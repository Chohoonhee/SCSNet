import math

import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)

def default_norm(n_feats):
    return nn.BatchNorm2d(n_feats)

def default_act():
    return nn.ReLU(True)

import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

def empty_h(x, n_feats):
    '''
        create an empty hidden state

        input
            x:      B x T x 3 x H x W

        output
            h:      B x C x H/4 x W/4
    '''
    b = x.size(0)
    h, w = x.size()[-2:]
    return x.new_zeros((b, n_feats, h//4, w//4))

class Normalization(nn.Conv2d):
    """Normalize input tensor value with convolutional layer"""
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        super(Normalization, self).__init__(3, 3, kernel_size=1)
        tensor_mean = torch.Tensor(mean)
        tensor_inv_std = torch.Tensor(std).reciprocal()

        self.weight.data = torch.eye(3).mul(tensor_inv_std).view(3, 3, 1, 1)
        self.bias.data = torch.Tensor(-tensor_mean.mul(tensor_inv_std))

        for params in self.parameters():
            params.requires_grad = False

class BasicBlock(nn.Sequential):
    """Convolution layer + Activation layer"""
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act):

        modules = []
        modules.append(
            conv(in_channels, out_channels, kernel_size, bias=bias))
        if norm: modules.append(norm(out_channels))
        if act: modules.append(act())

        super(BasicBlock, self).__init__(*modules)

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act):

        super(ResBlock, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm: modules.append(norm(n_feats))
            if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class ResBlock_mobile(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act, dropout=False):

        super(ResBlock_mobile, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=False, groups=n_feats))
            modules.append(conv(n_feats, n_feats, 1, bias=False))
            if dropout and i == 0: modules.append(nn.Dropout2d(dropout))
            if norm: modules.append(norm(n_feats))
            if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(
        self, scale, n_feats, bias=True,
        conv=default_conv, norm=False, act=False):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feats, 4 * n_feats, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if norm: modules.append(norm(n_feats))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feats, 9 * n_feats, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if norm: modules.append(norm(n_feats))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)

# Only support 1 / 2
class PixelSort(nn.Module):
    """The inverse operation of PixelShuffle
    Reduces the spatial resolution, increasing the number of channels.
    Currently, scale 0.5 is supported only.
    Later, torch.nn.functional.pixel_sort may be implemented.
    Reference:
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/modules/pixelshuffle.html#PixelShuffle
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/functional.html#pixel_shuffle
    """
    def __init__(self, upscale_factor=0.5):
        super(PixelSort, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, 2, 2, h // 2, w // 2)
        x = x.permute(0, 1, 5, 3, 2, 4).contiguous()
        x = x.view(b, 4 * c, h // 2, w // 2)

        return x

class Downsampler(nn.Sequential):
    def __init__(
        self, scale, n_feats, bias=True,
        conv=default_conv, norm=False, act=False):

        modules = []
        if scale == 0.5:
            modules.append(PixelSort())
            modules.append(conv(4 * n_feats, n_feats, 3, bias))
            if norm: modules.append(norm(n_feats))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*modules)

