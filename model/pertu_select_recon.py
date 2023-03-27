import torch
import torch.nn as nn

from . import common
from .ResNet import ResNet
# from .ResNet import ResNet_event
from data import common as cm
from model import perturbations
import math
import torch.nn.functional as F

class ResNet_event(nn.Module):
    def __init__(self, args, in_channels=18, out_channels=3, n_feats=None, kernel_size=None, n_resblocks=None, mean_shift=True):
        super(ResNet_event, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = args.n_feats if n_feats is None else n_feats
        # import pdb
        # pdb.set_trace()
        self.kernel_size = args.kernel_size if kernel_size is None else kernel_size
        self.n_resblocks = args.n_resblocks if n_resblocks is None else n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2

        modules = []
        modules.append(common.default_conv(self.in_channels, self.n_feats, self.kernel_size))
        for _ in range(self.n_resblocks):
            modules.append(common.ResBlock(self.n_feats, self.kernel_size))
        modules.append(common.default_conv(self.n_feats, self.out_channels, self.kernel_size))

        self.body = nn.Sequential(*modules)

    def forward(self, input):
        
        output = self.body(input)

        return output

def argtopk(x, axis=-1):
    _, index = torch.topk(x, k=3, dim=axis)
    
    # expand = torch.nn.functional.one_hot(index.squeeze())
    # print(expand.shape)
    # output = expand.float()

    return F.one_hot(index, list(x.shape)[axis]).float()

def argmax(x, axis=-1):
  	return F.one_hot(torch.argmax(x, dim=axis), list(x.shape)[axis]).float()


def build_model(args):
    return MSResNet_with_event(args)
    # return MSResNet(args)

class conv_end(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, ratio=2):
        super(conv_end, self).__init__()

        modules = [
            common.default_conv(in_channels, out_channels, kernel_size),
            nn.PixelShuffle(ratio)
        ]

        self.uppath = nn.Sequential(*modules)

    def forward(self, x):
        return self.uppath(x)

class MSResNet(nn.Module):
    def __init__(self, args):
        super(MSResNet, self).__init__()

        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2

        self.n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        self.kernel_size = args.kernel_size

        self.n_scales = args.n_scales

        self.body_models = nn.ModuleList([
            ResNet(args, 3, 3, mean_shift=False),
        ])
        for _ in range(1, self.n_scales):
            self.body_models.insert(0, ResNet(args, 6, 3, mean_shift=False))

        self.conv_end_models = nn.ModuleList([None])
        for _ in range(1, self.n_scales):
            self.conv_end_models += [conv_end(3, 12)]

        

    def forward(self, input_pyramid):

        scales = range(self.n_scales-1, -1, -1)    # 0: fine, 2: coarse

        for s in scales:
            input_pyramid[s] = input_pyramid[s] - self.mean

        output_pyramid = [None] * self.n_scales

        input_s = input_pyramid[-1]
        for s in scales:    # [2, 1, 0]
            output_pyramid[s] = self.body_models[s](input_s)
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                input_s = torch.cat((input_pyramid[s-1], up_feat), 1)

        for s in scales:
            output_pyramid[s] = output_pyramid[s] + self.mean

        return output_pyramid

class MSResNet_with_event(nn.Module):
    def __init__(self, args):
        super(MSResNet_with_event, self).__init__()

        self.rgb_range = args.rgb_range
        self.mean = self.rgb_range / 2

        self.n_resblocks = args.n_resblocks
        self.n_feats = args.n_feats
        self.kernel_size = args.kernel_size

        self.n_scales = args.n_scales

        


        self.encoding_with_image = nn.ModuleList()
        for s in range(self.n_scales):
            out_dim = 32*(2**s)
            self.encoding_with_image.append(ResNet_event(args, 8, mean_shift=False, n_resblocks = 7, n_feats = 16*(2**s), out_channels = out_dim))
        
        self.conv_score = []
        self.conv_score.append(nn.Conv2d(224, 128, kernel_size=3, padding=(3 // 2), stride = 2, bias=True))
        self.conv_score.append(nn.Conv2d(128, 32, kernel_size=3, padding=(3 // 2), stride = 2, bias=True))
        self.conv_score.append(nn.Conv2d(32, 5, kernel_size=3, padding=(3 // 2), stride = 2, bias=True))
        self.conv_score.append(nn.AvgPool2d(32))
        self.conv_score = nn.Sequential(*self.conv_score)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pert_argtopk = perturbations.perturbed(argtopk, 
                                        num_samples=200, 
                                        sigma=0.05, 
                                        noise='gumbel',
                                        batched=True,
                                        device=self.device)

        self.body_models = [ResNet_event(args, 3, 3, mean_shift=False)]
        self.body_models.append(ResNet_event(args, 3, 3, mean_shift=False))
        self.body_models = nn.Sequential(*self.body_models)



    def forward(self, input_pyramid):
        
        image_pyramid, event_pyramid = input_pyramid[0], input_pyramid[1]
        scales = range(self.n_scales-1, -1, -1)    # 0: fine, 2: coarse

        concat_pyramid = [None] * self.n_scales
        encoded_pyramid = [None] * self.n_scales
        for s in scales:
            concat_pyramid[s] = torch.cat((image_pyramid[s], event_pyramid[s]), 1)

        for s in range(self.n_scales):    # [2, 1, 0]
            input_s = concat_pyramid[s]
            encoded_pyramid[s] = self.encoding_with_image[s](input_s)
            if s > 0:
                upsample = nn.Upsample(scale_factor = 2**s, mode = 'bilinear')
                encoded_pyramid[s] = upsample(encoded_pyramid[s])
                encoded_feature = torch.cat((encoded_feature, encoded_pyramid[s]), 1)
            else:
                encoded_feature = encoded_pyramid[s]
        b, c, h, w = event_pyramid[0].shape
        score = self.conv_score(encoded_feature).squeeze(-1).squeeze(-1)
        if self.training:
            one_hot = self.pert_argtopk(score)
        else:
            one_hot = argtopk(score)

        
        
        # print("score shape")
        # print(score.shape)
        # print(score)
        # print("one hot shape")
        # print(one_hot.shape)
        # print(one_hot)
        
        input_event = event_pyramid[0].view(b, c, -1)

        select_event = torch.bmm(one_hot, input_event).view(b, -1, h, w)
        
        output = self.body_models(select_event)

        return output
