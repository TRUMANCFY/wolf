__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from wolf.flows.flow import Flow


class CondConv1x1Flow(Flow):
    def __init__(self, in_channels, h_channels, inverse=False):
        super(CondConv1x1Flow, self).__init__(inverse)
        self.in_channels = in_channels
        self.scale_proj = nn.Linear(h_channels, in_channels)
        self.shift_proj = nn.Linear(h_channels, in_channels)
        self.weight = Parameter(torch.Tensor(in_channels, in_channels))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()
    
    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())
    
    def get_weight(self, h, batch_size):
        scale_vec = self.scale_proj(h)
        shift_vec = self.shift_proj(h)

        # scale_vec_norm = torch.norm(scale_vec)
        # shift_vec_norm = torch.norm(shift_vec)

        scale_vec = torch.tanh(scale_vec).div(self.in_channels)
        shift_vec = torch.tanh(shift_vec).div(self.in_channels)

        scale_vec = 1 + scale_vec.view(batch_size, self.in_channels, 1)
        shift_vec = shift_vec.view(batch_size, 1, self.in_channels)

        weight = self.weight.repeat(batch_size, 1, 1)
        weight = weight * scale_vec + shift_vec

        return weight

    @overrides
    def forward(self, input, h):
        # [batch_size, in_channel, H, W]
        batch_size, channels, H, W = input.size()
        assert channels == self.in_channels, 'The input channel does not match!'
        assert h is not None, 'Please use Conv1x1Flow, instead of CondConv1x1Flow'
        
        weight = self.get_weight(h, batch_size)
        weight = weight.view(batch_size * self.in_channels, self.in_channels, 1, 1)

        input = input.contiguous().view(1, batch_size * self.in_channels, H, W)

        out = F.conv2d(input, weight, groups=batch_size, stride=1, bias=None)
        _, logdet = torch.slogdet(weight)

        out = out.view(batch_size, self.in_channels, H, W)
        # print('condconv1x1 forward')

        return out, logdet.mean().mul(H * W)

    @overrides
    def backward(self, input, h):
        batch_size, channels, H, W = input.size()

        weight = self.get_weight(h, batch_size)
        weight_inv = torch.inverse(weight)
        weight_inv = weight_inv.view(batch_size * self.in_channels, self.in_channels, 1, 1)
        
        input = input.contiguous().view(batch_size * self.in_channels, H, W)

        out = F.conv2d(input, weight_inv, groups=batch_size, stride=1, bias=None)
        _, logdet = torch.slogdet(weight_inv)
        out = out.view(batch_size, self.in_channels, H, W)

        return out, logdet.mean().mul(H * W)


    @overrides
    def init(self, data, h, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data, h)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}'.format(self.inverse, self.in_channels)

    @classmethod
    def from_params(cls, params: Dict) -> "CondConv1x1Flow":
        return CondConv1x1Flow(**params)


class Conv1x1Flow(Flow):
    def __init__(self, in_channels, inverse=False):
        super(Conv1x1Flow, self).__init__(inverse)
        self.in_channels = in_channels
        self.weight = Parameter(torch.Tensor(in_channels, in_channels))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        batch, channels, H, W = input.size()
        out = F.conv2d(input, self.weight.view(self.in_channels, self.in_channels, 1, 1))
        _, logdet = torch.slogdet(self.weight)
        return out, logdet.mul(H * W)

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        batch, channels, H, W = input.size()
        out = F.conv2d(input, self.weight_inv.view(self.in_channels, self.in_channels, 1, 1))
        _, logdet = torch.slogdet(self.weight_inv)
        return out, logdet.mul(H * W)

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}'.format(self.inverse, self.in_channels)

    @classmethod
    def from_params(cls, params: Dict) -> "Conv1x1Flow":
        return Conv1x1Flow(**params)


class InvertibleLinearFlow(Flow):
    def __init__(self, in_features, inverse=False):
        super(InvertibleLinearFlow, self).__init__(inverse)
        self.in_features = in_features
        self.weight = Parameter(torch.Tensor(in_features, in_features))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

    @overrides
    def forward(self, input: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor or None
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        # [batch, N1, N2, ..., in_features]
        out = F.linear(input, self.weight)
        _, logdet = torch.slogdet(self.weight)
        if dim > 2:
            num = np.prod(input.size()[1:-1]).astype(float) if mask is None else mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor or None
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        # [batch, N1, N2, ..., in_features]
        out = F.linear(input, self.weight_inv)
        _, logdet = torch.slogdet(self.weight_inv)
        if dim > 2:
            num = np.prod(input.size()[1:-1]).astype(float) if mask is None else mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    @overrides
    def init(self, data: torch.Tensor, mask: Union[torch.Tensor, None] = None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data, mask=mask)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.in_features)

    @classmethod
    def from_params(cls, params: Dict) -> "InvertibleLinearFlow":
        return InvertibleLinearFlow(**params)


class InvertibleMultiHeadFlow(Flow):
    @staticmethod
    def _get_heads(in_features):
        units = [32, 16, 8]
        for unit in units:
            if in_features % unit == 0:
                return in_features // unit
        assert in_features < 8, 'features={}'.format(in_features)
        return 1

    def __init__(self, in_features, heads=None, type='A', inverse=False):
        super(InvertibleMultiHeadFlow, self).__init__(inverse)
        self.in_features = in_features
        if heads is None:
            heads = InvertibleMultiHeadFlow._get_heads(in_features)
        self.heads = heads
        self.type = type
        assert in_features % heads == 0, 'features ({}) should be divided by heads ({})'.format(in_features, heads)
        assert type in ['A', 'B'], 'type should belong to [A, B]'
        self.weight = Parameter(torch.Tensor(in_features // heads, in_features // heads))
        self.register_buffer('weight_inv', self.weight.data.clone())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        self.sync()

    def sync(self):
        self.weight_inv.copy_(self.weight.data.inverse())

    @overrides
    def forward(self, input: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor or None
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        size = input.size()
        dim = input.dim()
        # [batch, N1, N2, ..., heads, in_features/ heads]
        if self.type == 'A':
            out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
        else:
            out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

        out = F.linear(out, self.weight)
        if self.type == 'B':
            out = out.transpose(-2, -1).contiguous()
        out = out.view(*size)

        _, logdet = torch.slogdet(self.weight)
        if dim > 2:
            num = np.prod(size[1:-1]).astype(float) if mask is None else mask.view(size[0], -1).sum(dim=1)
            num *= self.heads
            logdet = logdet * num
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., Nl, in_features]
            mask: Tensor or None
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_features], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        size = input.size()
        dim = input.dim()
        # [batch, N1, N2, ..., heads, in_features/ heads]
        if self.type == 'A':
            out = input.view(*size[:-1], self.heads, self.in_features // self.heads)
        else:
            out = input.view(*size[:-1], self.in_features // self.heads, self.heads).transpose(-2, -1)

        out = F.linear(out, self.weight_inv)
        if self.type == 'B':
            out = out.transpose(-2, -1).contiguous()
        out = out.view(*size)

        _, logdet = torch.slogdet(self.weight_inv)
        if dim > 2:
            num = np.prod(size[1:-1]).astype(float) if mask is None else mask.view(size[0], -1).sum(dim=1)
            num *= self.heads
            logdet = logdet * num
        return out, logdet

    @overrides
    def init(self, data: torch.Tensor, mask: Union[torch.Tensor, None] = None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data, mask)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_features={}, heads={}, type={}'.format(self.inverse, self.in_features, self.heads, self.type)

    @classmethod
    def from_params(cls, params: Dict) -> "InvertibleMultiHeadFlow":
        return InvertibleMultiHeadFlow(**params)


InvertibleLinearFlow.register('invertible_linear')
InvertibleMultiHeadFlow.register('invertible_multihead')
Conv1x1Flow.register('conv1x1')
CondConv1x1Flow.register('cond_conv1x1')
