from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from model.submodule import *
import math
from model import perturbations
from . import common
from .ResNet import ResNet
from model.affinity_module import *

class ResNet_event(nn.Module):
    def __init__(self, in_channels=18, out_channels=3, n_feats=32, kernel_size=3, n_resblocks=9, mean_shift=True):
        super(ResNet_event, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = n_feats
        # import pdb
        # pdb.set_trace()
        self.kernel_size = kernel_size
        self.n_resblocks =  n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = 255
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


class SizeAdapter(object):
    """Converts size of input to standard size.

    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return int(math.ceil(size / self._minimum_size) * self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.

        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = (
            self._closest_larger_multiple_of_minimum_size(height) - height)
        self._pixels_pad_to_width = (
            self._closest_larger_multiple_of_minimum_size(width) - width)
        return nn.ZeroPad2d((self._pixels_pad_to_width, 0,
                             self._pixels_pad_to_height, 0))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.

        The cropping is performed using values save by the "pad_input"
        method.
        """
        return network_output[..., self._pixels_pad_to_height:, self.
                              _pixels_pad_to_width:]



class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}

class event_feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(event_feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(128, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}



class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


affinity_settings = {}
affinity_settings['win_w'] = 3
affinity_settings['win_h'] = 3
affinity_settings['dilation'] = [1, 2, 4, 8]

class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)

            self.event_feature_extraction = event_feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)

                                            
        else:
            self.concat_channels = 0
            self.feature_extraction = event_image_feature_extraction(affinity_settings, 4, 'separate')
            # self.event_feature_extraction = event_feature_extraction(affinity_settings, 4, 'separate')

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self._size_adapter = SizeAdapter()
        self.gwc_fusion = nn.Sequential(convbn(640, 480, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(480, 320, kernel_size=3, padding=1, stride=1,
                                                    bias=False))
        self.concat_fusion = nn.Sequential(convbn(24, 24, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(24, 12, kernel_size=3, padding=1, stride=1,
                                                    bias=False))



        # self.conv_score = nn.Sequential(convbn(53, 32, 3, 2, 1, 1),
        #                                nn.ReLU(inplace=True),
        #                                convbn(32, 32, 3, 2, 1, 1),
        #                                nn.ReLU(inplace=True),
        #                                convbn(32, 64, 3, 1, 1, 1),
        #                                nn.ReLU(inplace=True),
        #                                convbn(64, 32, 3, 2, 1, 1),
        #                                nn.ReLU(inplace=True),
        #                                convbn(32, 5, 3, 1, 1, 1),
        #                                nn.ReLU(inplace=True),
        #                                nn.AvgPool2d(40, 48))
        self.conv_score = nn.Sequential(convbn(8, 64, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(64, 128, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 128, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(64, 5, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d(1))

        self.conv_score = nn.Sequential(*self.conv_score)

        self.pert_argtopk = perturbations.perturbed(argtopk, 
                                        num_samples=200, 
                                        sigma=0.05, 
                                        noise='gumbel',
                                        batched=True)

        self.body_models = [ResNet_event(3, 3, mean_shift=False)]
        self.body_models = nn.Sequential(*self.body_models)

        self.conv_lstm = common.ConvLSTM(input_dim=1,
                        hidden_dim=[16, 32, 32],
                        kernel_size=(3, 3),
                        num_layers=3,
                        batch_first=True,
                        bias=True,
                        return_all_layers=False)

        

    def forward(self, input):
        left, right, left_event, right_event = input


        # left = self._size_adapter.pad(left/255.0)
        # right = self._size_adapter.pad(right/255.0)
        # left = left/255.0
        # right = right/255.0

        # left_event = self._size_adapter.pad(left_event)
        # right_event = self._size_adapter.pad(right_event)

        # left_noise = self._size_adapter.pad(left_noise/255.0)
        # right_noise = self._size_adapter.pad(right_noise/255.0)

        b, c, h, w = left_event.shape

        left_score = self.conv_score(torch.cat((left_event, left), 1)).squeeze(-1).squeeze(-1)
        right_score = self.conv_score(torch.cat((right_event, right), 1)).squeeze(-1).squeeze(-1)

        if self.training:
            left_one_hot = self.pert_argtopk(left_score)
            right_one_hot = self.pert_argtopk(right_score)
        else:
            left_one_hot = argtopk(left_score)
            # print("left_one_hot")
            # print(left_one_hot)
            right_one_hot = argtopk(right_score)
            # print("right one hot")
            # print(right_one_hot)


        # import pdb; pdb.set_trace()
        left_event = torch.bmm(left_one_hot, left_event.view(b, c, -1)).view(b, -1, h, w)
        right_event = torch.bmm(right_one_hot, right_event.view(b, c, -1)).view(b, -1, h, w)
        


        # left_output = self._size_adapter.unpad(self.body_models(left_event))
        # right_output = self._size_adapter.unpad(self.body_models(right_event))
        left_output = self.body_models(left_event)
        right_output = self.body_models(right_event)


        

       

        left_event = left_event.unsqueeze(2)
        right_event = right_event.unsqueeze(2)


        left_event_emb = self.conv_lstm(left_event)[1][0][0]
        right_event_emb = self.conv_lstm(right_event)[1][0][0]


        # event_features_left = self.event_feature_extraction(left_event_emb)
        # event_features_right = self.event_feature_extraction(right_event_emb)
        
        features_left = self.feature_extraction(left, left_event_emb)
        features_right = self.feature_extraction(right, right_event_emb)

        # features_left["gwc_feature"] = self.gwc_fusion(torch.cat((features_left["gwc_feature"], event_features_left["gwc_feature"]),1))
        # features_right["gwc_feature"] = self.gwc_fusion(torch.cat((features_right["gwc_feature"], event_features_right["gwc_feature"]),1))

        # features_left["concat_feature"] = self.concat_fusion(torch.cat((features_left["concat_feature"], event_features_left["concat_feature"]), 1))
        # features_right["concat_feature"] = self.concat_fusion(torch.cat((features_right["concat_feature"], event_features_right["concat_feature"]), 1))

        # gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
        #                               self.num_groups)

        gwc_volume = build_gwc_volume(features_left, features_right, self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)

        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            # pred3 = self._size_adapter.unpad(pred3)
            # pred2 = self._size_adapter.unpad(pred2)
            # pred1 = self._size_adapter.unpad(pred1)
            # pred0 = self._size_adapter.unpad(pred0)

            return [pred0, pred1, pred2, pred3, left_output, right_output]
            # return [pred0, pred1, pred2, pred3]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            # pred3 = self._size_adapter.unpad(pred3)
            return [pred3]


def GwcNet_G(d):
    return GwcNet(d, use_concat_volume=False)


def GwcNet_GC(d=48):
    return GwcNet(d, use_concat_volume=True)

def Model(d=128):
    return GwcNet(d, use_concat_volume=False)