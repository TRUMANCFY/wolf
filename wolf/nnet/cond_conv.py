import torch
import torch.nn as nn
import torch.nn.functional as F

# the method refers to a Korean paper
# batch operation referes to https://github.com/pytorch/pytorch/issues/17983

class CondConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, h_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CondConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.scale_proj = nn.Linear(h_channels, out_channels)
        self.shift_proj = nn.Linear(h_channels, in_channels)
    
    def forward(self, input, h):
        """
        input: [batch_size, #channel, H, W]
        h: [batch_size, h_channels]
        """
        if h is None:
            return F.conv2d(input, self.weight, bias=self.bias, stride=self.stride, dilation=self.dilation, padding=self.padding, groups=self.groups)
        # print('inputshape1: ', input.shape)
        # calculate the scale and shift
        # [batch, out_channels]
        scale_vec = self.scale_proj(h)
        # [batch, in_channels]
        shift_vec = self.shift_proj(h)
        # [out_channels, in_channels, kernel_size, kernel_size]
        
        # get the dims
        batch_size = scale_vec.shape[0]
        out_channels, in_channels, kernel_size, kernel_size = self.weight.shape
        _, _, image_h, image_w = input.shape

        # [batch_size, out_channels, in_channels, kernel_size, kernel_size]
        weight = self.weight.repeat(batch_size, 1, 1, 1, 1)
        # print('weightshape1: ', weight.shape)
        scale_vec = scale_vec.view(batch_size, out_channels, 1, 1, 1)
        shift_vec = shift_vec.view(batch_size, 1, in_channels, 1, 1)
        weight = weight * scale_vec + shift_vec
        # print('weightshape2: ', weight.shape)
        # [batch_size * out_channels, in_channels, kernel_size, kernel_size]
        weight = weight.view(batch_size * out_channels, in_channels, kernel_size, kernel_size)
        # print('weightshape3: ', weight.shape)
        # print('inputshape2: ', input.shape)
        input = input.contiguous().view(1, batch_size * in_channels, image_h, image_w)

        out = F.conv2d(input, weight, bias=None, stride=self.stride, dilation=self.dilation, groups=batch_size, padding=self.padding)
        out = out.view(batch_size, out_channels, image_h, image_w)

        return out