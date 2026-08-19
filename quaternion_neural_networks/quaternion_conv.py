from typing import ClassVar

import torch
import torch.nn.functional as F
from torch._jit_internal import List, Optional
from torch.nn import Module
from torch.nn.modules.utils import _pair, _single, _triple
from torch.nn.parameter import Parameter

import quaternion_neural_networks.quaternion_init as init


def _repeat_tuple(t, n):
    r"""Repeat each element of `t` for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in t for _ in range(n))


class _QuatConvNd(Module):

    __constants__: ClassVar[list[str]] = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, scale, rotation, quaternion_format, padding_mode):

        super().__init__()

        if in_channels % 4 != 0:
            raise ValueError('in_channels must be divisible by 4')
        if out_channels % 4 != 0:
            raise ValueError('out_channels must be divisible by 4')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.quat_in_channels = in_channels // 4
        self.quat_out_channels = out_channels // 4

        if self.quat_in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if self.quat_out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = _repeat_tuple(self.padding, 2)
        self.scale = scale
        self.rotation = rotation
        self.quaternion_format = quaternion_format

        if transposed:
            self.r_weight = Parameter(torch.Tensor(self.quat_in_channels // groups, self.quat_out_channels, *kernel_size))
            self.i_weight = Parameter(torch.Tensor(self.quat_in_channels // groups, self.quat_out_channels, *kernel_size))
            self.j_weight = Parameter(torch.Tensor(self.quat_in_channels // groups, self.quat_out_channels, *kernel_size))
            self.k_weight = Parameter(torch.Tensor(self.quat_in_channels // groups, self.quat_out_channels, *kernel_size))
        else:
            self.r_weight = Parameter(torch.Tensor(self.quat_out_channels, self.quat_in_channels // groups, *kernel_size))
            self.i_weight = Parameter(torch.Tensor(self.quat_out_channels, self.quat_in_channels // groups, *kernel_size))
            self.j_weight = Parameter(torch.Tensor(self.quat_out_channels, self.quat_in_channels // groups, *kernel_size))
            self.k_weight = Parameter(torch.Tensor(self.quat_out_channels, self.quat_in_channels // groups, *kernel_size))

        if self.scale:
            self.scale_param = Parameter(torch.Tensor(self.quat_out_channels, self.quat_in_channels // groups, *kernel_size))
        else:
            self.register_parameter('scale_param', None)

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.quat_out_channels, self.quat_in_channels // groups, *kernel_size), requires_grad=False)
            self.weight = self.rot_weight
        else:
            self.register_parameter('zero_kernel', None)
            self.weight = self.norot_weight

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.r_weight.data, self.i_weight.data, self.j_weight.data, self.k_weight.data = init.qkaiming_chi_normal(self.r_weight,
                                                                                                                  a=0,
                                                                                                                  mode='fan_in',
                                                                                                                  nonlinearity='relu')

        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)

        if self.bias is not None:
            self.bias.data.zero_()

    def get_weight(self):
        if self.training:
            self._cached_weight = None
            return self.weight(self.r_weight, self.i_weight, self.j_weight, self.k_weight, scale_param=self.scale_param, zero_kernel=self.zero_kernel)
        else:
            if not hasattr(self, '_cached_weight') or self._cached_weight is None:
                self._cached_weight = self.weight(self.r_weight, self.i_weight, self.j_weight, self.k_weight, scale_param=self.scale_param, zero_kernel=self.zero_kernel)
            return self._cached_weight

    def rot_weight(self,  r_weight, i_weight, j_weight, k_weight, scale_param, zero_kernel):
        """
          Applies a quaternion rotation and convolution transformation to the incoming data:

            The rotation W*x*W^t can be replaced by R*x following:
            https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

            Works for unitary and non unitary weights.

            The initial size of the input must be a multiple of 3 if quaternion_format = False and
            4 if quaternion_format = True.
        """

        square_r = (r_weight*r_weight)
        square_i = (i_weight*i_weight)
        square_j = (j_weight*j_weight)
        square_k = (k_weight*k_weight)

        norm = torch.sqrt(square_r+square_i+square_j+square_k + 0.0001)

        r_n_weight = (r_weight / norm)
        i_n_weight = (i_weight / norm)
        j_n_weight = (j_weight / norm)
        k_n_weight = (k_weight / norm)

        norm_factor = 2.0

        square_i = norm_factor*(i_n_weight*i_n_weight)
        square_j = norm_factor*(j_n_weight*j_n_weight)
        square_k = norm_factor*(k_n_weight*k_n_weight)

        ri = (norm_factor*r_n_weight*i_n_weight)
        rj = (norm_factor*r_n_weight*j_n_weight)
        rk = (norm_factor*r_n_weight*k_n_weight)

        ij = (norm_factor*i_n_weight*j_n_weight)
        ik = (norm_factor*i_n_weight*k_n_weight)

        jk = (norm_factor*j_n_weight*k_n_weight)

        if self.quaternion_format:
            if self.scale:
                rot_kernel_1 = torch.cat([zero_kernel, scale_param * (1.0 - (square_j + square_k)), scale_param * (ij-rk), scale_param * (ik+rj)], dim=1)
                rot_kernel_2 = torch.cat([zero_kernel, scale_param * (ij+rk), scale_param * (1.0 - (square_i + square_k)), scale_param * (jk-ri)], dim=1)
                rot_kernel_3 = torch.cat([zero_kernel, scale_param * (ik-rj), scale_param * (jk+ri), scale_param * (1.0 - (square_i + square_j))], dim=1)
            else:
                rot_kernel_1 = torch.cat([zero_kernel, (1.0 - (square_j + square_k)), (ij-rk), (ik+rj)], dim=1)
                rot_kernel_2 = torch.cat([zero_kernel, (ij+rk), (1.0 - (square_i + square_k)), (jk-ri)], dim=1)
                rot_kernel_3 = torch.cat([zero_kernel, (ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=1)

            zero_kernel2 = torch.cat([zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=1)
            return torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)
        else:
            if self.scale:
                rot_kernel_1 = torch.cat([scale_param * (1.0 - (square_j + square_k)), scale_param * (ij-rk), scale_param * (ik+rj)], dim=1)
                rot_kernel_2 = torch.cat([scale_param * (ij+rk), scale_param * (1.0 - (square_i + square_k)), scale_param * (jk-ri)], dim=1)
                rot_kernel_3 = torch.cat([scale_param * (ik-rj), scale_param * (jk+ri), scale_param * (1.0 - (square_i + square_j))], dim=1)
            else:
                rot_kernel_1 = torch.cat([1.0 - (square_j + square_k), (ij-rk), (ik+rj)], dim=1)
                rot_kernel_2 = torch.cat([(ij+rk), 1.0 - (square_i + square_k), (jk-ri)], dim=1)
                rot_kernel_3 = torch.cat([(ik-rj), (jk+ri), (1.0 - (square_i + square_j))], dim=1)

            return torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    def norot_weight(self, r_weight, i_weight, j_weight, k_weight, *args, **kwargs):
        cat_kernels_4_r = torch.cat((r_weight, -i_weight, -j_weight, -k_weight), dim=1)
        cat_kernels_4_i = torch.cat((i_weight, r_weight, -k_weight, j_weight), dim=1)
        cat_kernels_4_j = torch.cat((j_weight, k_weight, r_weight, -i_weight), dim=1)
        cat_kernels_4_k = torch.cat((k_weight, -j_weight, i_weight, r_weight), dim=1)
        return torch.cat((cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k), dim=0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.scale:
            s += ', scale={scale}'
        if self.rotation:
            s += ', rotation={rotation}'
        if self.quaternion_format:
            s += ', quaternion_format={quaternion_format}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class QuatConv1d(_QuatConvNd):
    r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image. This should be divisible by 4
        out_channels (int): Number of channels produced by the convolution. This should be divisible by 4
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

    Examples::

        >>> m = QuatConv1d(4, 16, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, scale=False, rotation=False, quaternion_format=False, padding_mode='zeros'):

        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, scale, rotation, quaternion_format, padding_mode)

    def forward(self, input):
        weight = self.get_weight()

        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride, _single(0), self.dilation, self.groups)

        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuatConv2d(_QuatConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image. This should be divisible by 4
        out_channels (int): Number of channels produced by the convolution. This should be divisible by 4
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, scale=False, rotation=False, quaternion_format=False, padding_mode='zeros'):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, scale, rotation, quaternion_format, padding_mode)

    def forward(self, input):
        weight = self.get_weight()

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuatConv3d(_QuatConvNd):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image. This should be divisible by 4
        out_channels (int): Number of channels produced by the convolution. This should be divisible by 4
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, scale=False, rotation=False, quaternion_format=False, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias, scale, rotation, quaternion_format, padding_mode)

    def forward(self, input):
        weight = self.get_weight()

        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride, _triple(0), self.dilation, self.groups)

        return F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class _QuatConvTransposeNd(_QuatConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias,  scale, rotation, quaternion_format, padding_mode):
        if padding_mode != 'zeros':
            raise ValueError(f'Only "zeros" padding mode is supported for {self.__class__.__name__}')

        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, scale, rotation, quaternion_format, padding_mode)

    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    f"output_size must have {k} or {k + 2} elements (got {len(output_size)})"
                    )

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] + kernel_size[d])
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range "
                        f"from {min_sizes} to {max_sizes} (for an input of {input.size()[2:]})")

            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class QuatConvTranspose1d(_QuatConvTransposeNd):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation}
                        \times (\text{kernel\_size} - 1) + \text{output\_padding} + 1

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels).
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \text{kernel\_size}}`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1,  scale=False, rotation=False, quaternion_format=False, padding_mode='zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, scale, rotation, quaternion_format, padding_mode)

    def forward(self, input, output_size=None):

        weight = self.get_weight()
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose1d(input, weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)


class QuatConvTranspose2d(_QuatConvTransposeNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

        .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
        .. math::
              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1,  scale=False, rotation=False, quaternion_format=False, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, scale, rotation, quaternion_format, padding_mode)

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        weight = self.get_weight()
        return F.conv_transpose2d(input, weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)


class QuatConvTranspose3d(_QuatConvTransposeNd):
    r"""Applies a 3D transposed convolution operator over an input image composed of several input
    planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

        .. math::
              D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
        .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1
        .. math::
              W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
                        \times (\text{kernel\_size}[2] - 1) + \text{output\_padding}[2] + 1


    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1,  scale=False, rotation=False, quaternion_format=False, padding_mode='zeros'):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, scale, rotation, quaternion_format, padding_mode)

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose3d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        weight = self.get_weight()

        return F.conv_transpose3d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

