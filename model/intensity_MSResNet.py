import torch
import torch.nn as nn

from . import common
from .ResNet import ResNet
from .ResNet import ResNet_event
from data import common as cm

import math

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

        self.body_models = nn.ModuleList([
            ResNet_event(args, 8, 3, mean_shift=False),
        ])
        for _ in range(1, self.n_scales):
            self.body_models.insert(0, ResNet_event(args, 11, 3, mean_shift=False))

        self.conv_end_models = nn.ModuleList([None])
        for _ in range(1, self.n_scales):
            self.conv_end_models += [conv_end(3, 12)]

        # self.size_adaptor = SizeAdapter()

    # def forward(self, event_pyramid, input_pyramid):
    def forward(self, input_pyramid):
        
        event_pyramid, input_pyramid, mode, n_scales = input_pyramid[1], input_pyramid[0], input_pyramid[2], input_pyramid[3]

        # import pdb
        # pdb.set_trace()
        scales = range(self.n_scales-1, -1, -1)    # 0: fine, 2: coarse

        for s in scales:
            input_pyramid[s] = input_pyramid[s] - self.mean

        output_pyramid = [None] * self.n_scales

        for s in scales:
            input_pyramid[s] = torch.cat((input_pyramid[s], event_pyramid[s]), 1)

        input_s = input_pyramid[-1]
        for s in scales:    # [2, 1, 0]
            # import pdb
            # pdb.set_trace()
            output_pyramid[s] = self.body_models[s](input_s)
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                # import pdb
                # pdb.set_trace()
                input_s = torch.cat((input_pyramid[s-1], up_feat), 1)

        for s in scales:
            output_pyramid[s] = output_pyramid[s] + self.mean

        return output_pyramid
