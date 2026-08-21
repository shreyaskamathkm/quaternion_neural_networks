"""
Quaternion Linear Layers.

Provides PyTorch modules for quaternion-based fully connected (linear) layers,
including implementations with custom autograd functions for memory efficiency
and rotational quaternion variants.
"""


import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

import quaternion_neural_networks.quaternion_init as init


def quaternion_linear(input: Tensor, r_weight: Tensor, i_weight: Tensor, j_weight: Tensor, k_weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Applies a quaternion linear transformation to the incoming data.

    The forward phase of a QNN is defined as W * Inputs (with * equal to the Hamilton product).
    The constructed cat_kernels_4_quaternion is a modified version of the quaternion representation
    so when we do torch.mm(Input,W) it's equivalent to W * Inputs.

    Args:
        input (Tensor): The input data.
        r_weight (Tensor): Real weight tensor.
        i_weight (Tensor): First imaginary weight tensor.
        j_weight (Tensor): Second imaginary weight tensor.
        k_weight (Tensor): Third imaginary weight tensor.
        bias (Optional[Tensor], optional): Bias tensor. Defaults to None.

    Returns:
        Tensor: The output of the linear transformation.
    """
    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=0)
    cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=0)
    cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=0)
    cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=0)
    cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)

    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, cat_kernels_4_quaternion)
        else:
            return torch.mm(input, cat_kernels_4_quaternion)
    else:
        output = torch.matmul(input, cat_kernels_4_quaternion)
        if bias is not None:
            return output + bias
        else:
            return output


def quaternion_linear_rotation(input: Tensor, zero_kernel: Tensor, r_weight: Tensor, i_weight: Tensor, j_weight: Tensor, k_weight: Tensor, bias: Tensor | None = None, quaternion_format: bool = False, scale: Tensor | None = None) -> Tensor:
    """Applies a quaternion rotation transformation to the incoming data.

    The rotation W*x*W^t can be replaced by R*x following:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Args:
        input (Tensor): The input data.
        zero_kernel (Tensor): Zero kernel tensor.
        r_weight (Tensor): Real weight tensor.
        i_weight (Tensor): First imaginary weight tensor.
        j_weight (Tensor): Second imaginary weight tensor.
        k_weight (Tensor): Third imaginary weight tensor.
        bias (Optional[Tensor], optional): Bias tensor. Defaults to None.
        quaternion_format (bool, optional): If True, input size must be multiple of 4, else 3. Defaults to False.
        scale (Optional[Tensor], optional): Optional scaling factor. Defaults to None.

    Returns:
        Tensor: The output of the rotation transformation.
    """
    square_r = (r_weight * r_weight)
    square_i = (i_weight * i_weight)
    square_j = (j_weight * j_weight)
    square_k = (k_weight * k_weight)

    norm = torch.sqrt(square_r + square_i + square_j + square_k + 0.0001)

    r_n_weight = (r_weight / norm)
    i_n_weight = (i_weight / norm)
    j_n_weight = (j_weight / norm)
    k_n_weight = (k_weight / norm)

    norm_factor = 2.0

    square_i = norm_factor * (i_n_weight * i_n_weight)
    square_j = norm_factor * (j_n_weight * j_n_weight)
    square_k = norm_factor * (k_n_weight * k_n_weight)

    ri = (norm_factor * r_n_weight * i_n_weight)
    rj = (norm_factor * r_n_weight * j_n_weight)
    rk = (norm_factor * r_n_weight * k_n_weight)

    ij = (norm_factor * i_n_weight * j_n_weight)
    ik = (norm_factor * i_n_weight * k_n_weight)

    jk = (norm_factor * j_n_weight * k_n_weight)

    if quaternion_format:
        if scale is not None:
            rot_kernel_1 = torch.cat([zero_kernel, scale * (1.0 - (square_j + square_k)), scale * (ij - rk), scale * (ik + rj)], dim=0)
            rot_kernel_2 = torch.cat([zero_kernel, scale * (ij + rk), scale * (1.0 - (square_i + square_k)), scale * (jk - ri)], dim=0)
            rot_kernel_3 = torch.cat([zero_kernel, scale * (ik - rj), scale * (jk + ri), scale * (1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1 = torch.cat([zero_kernel, (1.0 - (square_j + square_k)), (ij - rk), (ik + rj)], dim=0)
            rot_kernel_2 = torch.cat([zero_kernel, (ij + rk), (1.0 - (square_i + square_k)), (jk - ri)], dim=0)
            rot_kernel_3 = torch.cat([zero_kernel, (ik - rj), (jk + ri), (1.0 - (square_i + square_j))], dim=0)

        zero_kernel2 = torch.cat([zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=0)
        global_rot_kernel = torch.cat([zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1)

    else:
        if scale is not None:
            rot_kernel_1 = torch.cat([scale * (1.0 - (square_j + square_k)), scale * (ij - rk), scale * (ik + rj)], dim=0)
            rot_kernel_2 = torch.cat([scale * (ij + rk), scale * (1.0 - (square_i + square_k)), scale * (jk - ri)], dim=0)
            rot_kernel_3 = torch.cat([scale * (ik - rj), scale * (jk + ri), scale * (1.0 - (square_i + square_j))], dim=0)
        else:
            rot_kernel_1 = torch.cat([1.0 - (square_j + square_k), (ij - rk), (ik + rj)], dim=0)
            rot_kernel_2 = torch.cat([(ij + rk), 1.0 - (square_i + square_k), (jk - ri)], dim=0)
            rot_kernel_3 = torch.cat([(ik - rj), (jk + ri), (1.0 - (square_i + square_j))], dim=0)

        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=1)

    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, global_rot_kernel)
        else:
            return torch.mm(input, global_rot_kernel)
    else:
        output = torch.matmul(input, global_rot_kernel)
        if bias is not None:
            return output + bias
        else:
            return output


class QuaternionLinearAutograd(Module):
    """Applies a quaternion linear transformation to the incoming data with custom autograd.

    A custom Autograd function is called to drastically reduce the VRAM consumption.
    Nonetheless, computing time is also slower compared to QuaternionLinear().

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        seed (Optional[int], optional): Random seed. Defaults to None.
        rotation (bool, optional): Use rotation. Defaults to False.
        quaternion_format (bool, optional): Use quaternion format. Defaults to False.

    Shape:
        - Input: `(*, H_in)` where `*` means any number of dimensions including none and `H_in = in_features`.
        - Output: `(*, H_out)` where all but the last dimension are the same shape as the input and `H_out = out_features`.

    Examples::

        >>> m = QuaternionLinearAutograd(20, 32)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 32])
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: int | None = None, rotation: bool = False, quaternion_format: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quat_in_features = in_features // 4
        self.quat_out_features = out_features // 4
        self.rotation = rotation
        self.quaternion_format = quaternion_format

        self.r_weight = Parameter(torch.Tensor(self.quat_in_features, self.quat_out_features))
        self.i_weight = Parameter(torch.Tensor(self.quat_in_features, self.quat_out_features))
        self.j_weight = Parameter(torch.Tensor(self.quat_in_features, self.quat_out_features))
        self.k_weight = Parameter(torch.Tensor(self.quat_in_features, self.quat_out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.quat_out_features * 4))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.r_weight.data, self.i_weight.data, self.j_weight.data, self.k_weight.data = init.qkaiming_chi_normal(
            self.r_weight, a=0, mode='fan_in', nonlinearity='relu'
        )
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the QuaternionLinearAutograd module.

        Args:
            input (Tensor): Input data.

        Returns:
            Tensor: Output data.
        """
        if self.rotation:
            return quaternion_linear_rotation(input, getattr(self, 'zero_kernel', None), self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias, self.quaternion_format, getattr(self, 'scale_param', None))  # type: ignore
        else:
            return quaternion_linear(
                input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias
            )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None})")


class QuaternionLinear(Module):
    """Applies a quaternion linear transformation to the incoming data.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
        init_criterion (str, optional): Initialization criterion. Defaults to 'glorot'.
        weight_init (str, optional): Weight initialization method. Defaults to 'quaternion'.
        seed (Optional[int], optional): Random seed. Defaults to None.

    Shape:
        - Input: `(*, H_in)` where `*` means any number of dimensions including none and `H_in = in_features`.
        - Output: `(*, H_out)` where all but the last dimension are the same shape as the input and `H_out = out_features`.

    Examples::

        >>> m = QuaternionLinear(20, 32)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 32])
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_criterion: str = 'glorot', weight_init: str = 'quaternion', seed: int | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quat_in_features = in_features // 4
        self.quat_out_features = out_features // 4

        self.r_weight = Parameter(torch.Tensor(self.quat_in_features, self.quat_out_features))
        self.i_weight = Parameter(torch.Tensor(self.quat_in_features, self.quat_out_features))
        self.j_weight = Parameter(torch.Tensor(self.quat_in_features, self.quat_out_features))
        self.k_weight = Parameter(torch.Tensor(self.quat_in_features, self.quat_out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(self.quat_out_features * 4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.r_weight.data, self.i_weight.data, self.j_weight.data, self.k_weight.data = init.qkaiming_chi_normal(
            self.r_weight, a=0, mode='fan_in', nonlinearity='relu'
        )
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the QuaternionLinear module.

        Args:
            input (Tensor): Input data.

        Returns:
            Tensor: Output data.
        """
        if input.dim() == 3:
            T, N, C = input.size()
            input_reshaped = input.view(T * N, C)
        else:
            input_reshaped = input

        r, i, j, k = torch.chunk(input_reshaped, 4, dim=1)
        r_w, i_w, j_w, k_w = self.r_weight, self.i_weight, self.j_weight, self.k_weight

        y_r = r.mm(r_w) - i.mm(i_w) - j.mm(j_w) - k.mm(k_w)
        y_i = r.mm(i_w) + i.mm(r_w) + j.mm(k_w) - k.mm(j_w)
        y_j = r.mm(j_w) - i.mm(k_w) + j.mm(r_w) + k.mm(i_w)
        y_k = r.mm(k_w) + i.mm(j_w) - j.mm(i_w) + k.mm(r_w)

        output = torch.cat([y_r, y_i, y_j, y_k], dim=1)
        if self.bias is not None:
            output += self.bias

        if input.dim() == 3:
            output = output.view(T, N, output.size(1))

        return output

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None}, "
                f"init_criterion={self.init_criterion}, weight_init={self.weight_init}, "
                f"seed={self.seed})")
