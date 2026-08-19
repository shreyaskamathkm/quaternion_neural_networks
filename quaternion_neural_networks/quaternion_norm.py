from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class QuaternionGroupNorm2d(nn.Module):
    r"""Applies Quaternion Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization`_ .

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    __constants__: ClassVar[list[str]] = ['num_groups', 'num_channels', 'eps', 'affine', 'weight',
                     'bias']

    def __init__(self, groups_per_quat, num_channels, eps=1e-5, affine=True):
        super().__init__()

        assert num_channels % 4 == 0, 'The number_features should be divisible by 4'
        q_channels = num_channels//4
        # if num_groups > qfeatures//2:
        assert q_channels % groups_per_quat == 0, f'The groups per quat should be divisible by {q_channels}'
        self.num_groups = groups_per_quat * 4
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class QuaternionBatchNorm2d(nn.Module):
    r"""Applies a 2D Quaternion Batch Normalization by using a single variance across the incoming data.
        """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()

        self.num_features = num_features // 4
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, 4, self.num_features, 1, 1))
            self.register_buffer('running_var', torch.ones(1, 1, self.num_features, 1, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f'expected 4D input (got {input.dim()}D input)')

    def forward(self, input):
        self._check_input_dim(input)

        N, C, H, W = input.shape
        input = input.view(N, 4, C // 4, H, W)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.copy_( self.num_batches_tracked + 1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        if self.training:
            mean = input.mean([0, 3, 4], keepdim=True)
            var = ((input - mean) ** 2).mean([0, 1, 3, 4], keepdim=True)
            n = input.numel() / input.size(2)

            with torch.no_grad():
                self.running_mean.copy_(exponential_average_factor * mean + (1 - exponential_average_factor) * self.running_mean)
                # update running_var with unbiased var
                self.running_var.copy_(exponential_average_factor * var * n / (n - 1) + (1 - exponential_average_factor) * self.running_var)

        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean) * torch.rsqrt(var + self.eps)

        # Convert it back
        input = input.view(N, C, H, W)
        if self.affine:
            return input * self.weight + self.bias
        else:
            return input

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

