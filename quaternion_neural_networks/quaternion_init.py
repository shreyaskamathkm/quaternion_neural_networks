"""
Quaternion weight initialization operations.

Provides custom initialization functions tailored for quaternion weights,
such as Kaiming and Xavier initialization for quaternions.
"""

import math

import numpy as np
import torch
from scipy.stats import chi
from torch import Tensor
from torch.nn.init import _calculate_correct_fan, _calculate_fan_in_and_fan_out, calculate_gain


def qxavier_chi_normal(tensor: Tensor, gain: float = 1.) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initializes a quaternion tensor using Xavier initialization with chi distribution.

    Args:
        tensor (Tensor): An n-dimensional tensor.
        gain (float, optional): An optional scaling factor. Defaults to 1.0.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Four initialized tensors representing
            the real and three imaginary parts of the quaternion.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1 / 2.0 * float(fan_in + fan_out))

    modulus = torch.from_numpy(chi.rvs(4, loc=-std, scale=std / 2, size=tensor.shape))
    phase = torch.empty(tensor.shape).uniform_(-np.pi, np.pi)

    # Purely imaginary quaternions unitary
    with torch.no_grad():
        i = torch.empty(tensor.shape).uniform_(-1, 1.)
        j = torch.empty(tensor.shape).uniform_(-1, 1.)
        k = torch.empty(tensor.shape).uniform_(-1, 1.)
        norm = torch.sqrt(i.pow(2) + j.pow(2) + k.pow(2) + torch.finfo(tensor.dtype).eps)
        i /= norm
        j /= norm
        k /= norm

    weight_r = modulus * torch.cos(phase)
    weight_i = modulus * torch.sin(phase) * i
    weight_j = modulus * torch.sin(phase) * j
    weight_k = modulus * torch.sin(phase) * k

    return (
        weight_r.type_as(tensor.data),
        weight_i.type_as(tensor.data),
        weight_j.type_as(tensor.data),
        weight_k.type_as(tensor.data)
    )


def qkaiming_chi_normal(tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initializes a quaternion tensor using Kaiming initialization with chi distribution.

    Args:
        tensor (Tensor): An n-dimensional tensor.
        a (float, optional): The negative slope of the rectifier used after this layer. Defaults to 0.
        mode (str, optional): Either 'fan_in' or 'fan_out'. Defaults to 'fan_in'.
        nonlinearity (str, optional): The non-linear function name. Defaults to 'leaky_relu'.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Four initialized tensors representing
            the real and three imaginary parts of the quaternion.
    """
    fan = _calculate_correct_fan(tensor, mode)  # type: ignore
    gain = calculate_gain(nonlinearity, a)  # type: ignore
    std = gain / math.sqrt(2 * fan)

    modulus = torch.from_numpy(chi.rvs(4, loc=0, scale=std, size=tensor.shape))
    phase = torch.empty(tensor.shape).uniform_(-np.pi, np.pi)

    # Purely imaginary quaternions unitary
    with torch.no_grad():
        i = torch.empty(tensor.shape).uniform_(-1, 1.)
        j = torch.empty(tensor.shape).uniform_(-1, 1.)
        k = torch.empty(tensor.shape).uniform_(-1, 1.)

        norm = torch.sqrt(i.pow(2) + j.pow(2) + k.pow(2) + torch.finfo(tensor.dtype).eps)
        i /= norm
        j /= norm
        k /= norm

    weight_r = modulus * torch.cos(phase)
    weight_i = modulus * i * torch.sin(phase)
    weight_j = modulus * j * torch.sin(phase)
    weight_k = modulus * k * torch.sin(phase)

    return (
        weight_r.type_as(tensor.data),
        weight_i.type_as(tensor.data),
        weight_j.type_as(tensor.data),
        weight_k.type_as(tensor.data)
    )


def qxavier_uniform(tensor: Tensor, gain: float = 1.) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initializes a quaternion tensor using Xavier uniform initialization.

    Args:
        tensor (Tensor): An n-dimensional tensor.
        gain (float, optional): An optional scaling factor. Defaults to 1.0.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Four initialized tensors representing
            the real and three imaginary parts of the quaternion.
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1 / 2.0 * float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    modulus = torch.empty(tensor.shape).uniform_(-a, a)
    phase = torch.empty(tensor.shape).uniform_(-np.pi, np.pi)

    # Purely imaginary quaternions unitary
    with torch.no_grad():
        i = torch.empty(tensor.shape).uniform_(-a, a)
        j = torch.empty(tensor.shape).uniform_(-a, a)
        k = torch.empty(tensor.shape).uniform_(-a, a)
        norm = torch.sqrt(i.pow(2) + j.pow(2) + k.pow(2) + torch.finfo(tensor.dtype).eps)
        i /= norm
        j /= norm
        k /= norm

    weight_r = modulus * torch.cos(phase)
    weight_i = modulus * torch.sin(phase) * i
    weight_j = modulus * torch.sin(phase) * j
    weight_k = modulus * torch.sin(phase) * k

    return (
        weight_r.type_as(tensor.data),
        weight_i.type_as(tensor.data),
        weight_j.type_as(tensor.data),
        weight_k.type_as(tensor.data)
    )


def qkaiming_uniform(tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Initializes a quaternion tensor using Kaiming uniform initialization.

    Args:
        tensor (Tensor): An n-dimensional tensor.
        a (float, optional): The negative slope of the rectifier used after this layer. Defaults to 0.
        mode (str, optional): Either 'fan_in' or 'fan_out'. Defaults to 'fan_in'.
        nonlinearity (str, optional): The non-linear function name. Defaults to 'leaky_relu'.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Four initialized tensors representing
            the real and three imaginary parts of the quaternion.
    """
    fan = _calculate_correct_fan(tensor, mode)  # type: ignore
    gain = calculate_gain(nonlinearity, a)  # type: ignore
    std = gain / math.sqrt(2 * fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    with torch.no_grad():
        i = torch.empty(tensor.shape).uniform_(-bound, bound)
        j = torch.empty(tensor.shape).uniform_(-bound, bound)
        k = torch.empty(tensor.shape).uniform_(-bound, bound)
        norm = torch.sqrt(i.pow(2) + j.pow(2) + k.pow(2) + torch.finfo(tensor.dtype).eps)
        i /= norm
        j /= norm
        k /= norm

    modulus = torch.empty(tensor.shape).uniform_(-bound, bound)
    phase = torch.empty(tensor.shape).uniform_(-np.pi, np.pi)
    r = modulus * torch.cos(phase)
    i_part = modulus * i * torch.sin(phase)
    j_part = modulus * j * torch.sin(phase)
    k_part = modulus * k * torch.sin(phase)

    return r.type_as(tensor.data), i_part.type_as(tensor.data), j_part.type_as(tensor.data), k_part.type_as(tensor.data)


def get_upsample_filter(size: int) -> Tensor:
    """Make a 2D bilinear kernel suitable for upsampling.

    Args:
        size (int): Size of the kernel.

    Returns:
        Tensor: A 2D bilinear filter tensor.
    """
    factor = (size + 1) // 2
    center = factor - 1 if size % 2 == 1 else factor - 0.5
    og = np.ogrid[:size, :size]
    filter_matrix = (1 - abs(og[0] - center) / factor) * \
                    (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter_matrix).float()
