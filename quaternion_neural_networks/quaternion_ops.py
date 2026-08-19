"""
Quaternion mathematical operations and utilities.

Provides basic operations for quaternion tensors, such as extracting
components (r, i, j, k), concatenation, normalization, Hamilton product,
and exponentiation.
"""

import torch
from torch import Tensor


def q_normalize(input: Tensor, channel: int = 1) -> Tensor:
    """Normalizes a quaternion tensor.

    Args:
        input (Tensor): The input quaternion tensor.
        channel (int, optional): The dimension corresponding to the quaternion
            components. Defaults to 1.

    Returns:
        Tensor: A normalized quaternion tensor.
    """
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)

    norm = torch.sqrt(r * r + i * i + j * j + k * k + 0.0001)
    r = r / norm
    i = i / norm
    j = j / norm
    k = k / norm

    return torch.cat([r, i, j, k], dim=channel)


def check_input(input: Tensor) -> None:
    """Checks the validity of the input tensor for quaternion operations.

    Ensures the input tensor has valid dimensions (2 to 5) and that the
    feature dimension is divisible by 4.

    Args:
        input (Tensor): The input tensor to validate.

    Raises:
        RuntimeError: If the tensor dimensions are not between 2 and 5,
            or if the feature dimension is not divisible by 4.
    """
    if input.dim() not in {2, 3, 4, 5}:
        raise RuntimeError(
            "Quaternion linear accepts only input of dimension 2 or 3. "
            "Quaternion conv accepts up to 5 dim. input.dim = " + str(input.dim())
        )

    nb_hidden = input.size()[-1] if input.dim() < 4 else input.size()[1]

    if nb_hidden % 4 != 0:
        raise RuntimeError(
            "Quaternion Tensors must be divisible by 4. "
            "input.size()[1] = " + str(nb_hidden)
        )


def _get_feature_dim(input: Tensor) -> int:
    """Returns the feature dimension index for the input tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        int: The feature dimension index.
    """
    return input.dim() - 1 if input.dim() < 4 else 1


def get_r(input: Tensor) -> Tensor:
    """Extracts the real component (r) from a quaternion tensor.

    Args:
        input (Tensor): The input quaternion tensor.

    Returns:
        Tensor: The real component tensor.
    """
    check_input(input)
    r, _, _, _ = torch.chunk(input, 4, dim=_get_feature_dim(input))
    return r


def get_i(input: Tensor) -> Tensor:
    """Extracts the first imaginary component (i) from a quaternion tensor.

    Args:
        input (Tensor): The input quaternion tensor.

    Returns:
        Tensor: The first imaginary component tensor.
    """
    check_input(input)
    _, i, _, _ = torch.chunk(input, 4, dim=_get_feature_dim(input))
    return i


def get_j(input: Tensor) -> Tensor:
    """Extracts the second imaginary component (j) from a quaternion tensor.

    Args:
        input (Tensor): The input quaternion tensor.

    Returns:
        Tensor: The second imaginary component tensor.
    """
    check_input(input)
    _, _, j, _ = torch.chunk(input, 4, dim=_get_feature_dim(input))
    return j


def get_k(input: Tensor) -> Tensor:
    """Extracts the third imaginary component (k) from a quaternion tensor.

    Args:
        input (Tensor): The input quaternion tensor.

    Returns:
        Tensor: The third imaginary component tensor.
    """
    check_input(input)
    _, _, _, k = torch.chunk(input, 4, dim=_get_feature_dim(input))
    return k


def get_modulus(input: Tensor, vector_form: bool = False) -> Tensor:
    """Computes the modulus of a quaternion tensor.

    Args:
        input (Tensor): The input quaternion tensor.
        vector_form (bool, optional): Whether to keep the output in vector form
            or sum across the 0th dimension. Defaults to False.

    Returns:
        Tensor: The modulus tensor.
    """
    check_input(input)
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)
    if vector_form:
        return torch.sqrt(r * r + i * i + j * j + k * k)
    else:
        return torch.sqrt((r * r + i * i + j * j + k * k).sum(dim=0))


def get_normalized(input: Tensor, eps: float = 0.0001) -> Tensor:
    """Returns a normalized version of the quaternion tensor.

    Args:
        input (Tensor): The input quaternion tensor.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 0.0001.

    Returns:
        Tensor: The normalized quaternion tensor.
    """
    check_input(input)
    data_modulus = get_modulus(input)
    if input.dim() == 2:
        data_modulus_repeated = data_modulus.repeat(1, 4)
    elif input.dim() == 3:
        data_modulus_repeated = data_modulus.repeat(1, 1, 4)
    else:
        # Fallback for higher dimensions
        repeats = [1] * input.dim()
        repeats[_get_feature_dim(input)] = 4
        data_modulus_repeated = data_modulus.repeat(*repeats)

    return input / (data_modulus_repeated.expand_as(input) + eps)


def quaternion_exp(input: Tensor) -> Tensor:
    """Computes the quaternion exponential.

    Args:
        input (Tensor): The input quaternion tensor.

    Returns:
        Tensor: The exponentiated quaternion tensor.
    """
    r = get_r(input)
    i = get_i(input)
    j = get_j(input)
    k = get_k(input)

    norm_v = torch.sqrt(i * i + j * j + k * k) + 0.0001
    exp_val = torch.exp(r)

    r_out = torch.cos(norm_v)
    i_out = (i / norm_v) * torch.sin(norm_v)
    j_out = (j / norm_v) * torch.sin(norm_v)
    k_out = (k / norm_v) * torch.sin(norm_v)

    return torch.cat([exp_val * r_out, exp_val * i_out, exp_val * j_out, exp_val * k_out], dim=1)


def quaternion_concat(x: Tensor, y: Tensor) -> Tensor:
    """Concatenates two quaternion tensors along their components.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.

    Returns:
        Tensor: The concatenated quaternion tensor.
    """
    o_r_concat = torch.cat([get_r(x), get_r(y)], dim=1)
    o_i_concat = torch.cat([get_i(x), get_i(y)], dim=1)
    o_j_concat = torch.cat([get_j(x), get_j(y)], dim=1)
    o_k_concat = torch.cat([get_k(x), get_k(y)], dim=1)
    return torch.cat([o_r_concat, o_i_concat, o_j_concat, o_k_concat], dim=1)


def quaternion_concat_3(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Concatenates three quaternion tensors along their components.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.
        z (Tensor): The third input tensor.

    Returns:
        Tensor: The concatenated quaternion tensor.
    """
    o_r_concat = torch.cat([get_r(x), get_r(y), get_r(z)], dim=1)
    o_i_concat = torch.cat([get_i(x), get_i(y), get_i(z)], dim=1)
    o_j_concat = torch.cat([get_j(x), get_j(y), get_j(z)], dim=1)
    o_k_concat = torch.cat([get_k(x), get_k(y), get_k(z)], dim=1)
    return torch.cat([o_r_concat, o_i_concat, o_j_concat, o_k_concat], dim=1)


def hamilton_product(q0: Tensor, q1: Tensor) -> Tensor:
    """Applies a Hamilton product q0 * q1.

    Args:
        q0 (Tensor): The first quaternion tensor of shape (batch_size, quaternion_number).
        q1 (Tensor): The second quaternion tensor of shape (batch_size, quaternion_number).

    Returns:
        Tensor: The result of the Hamilton product.
    """
    q1_r = get_r(q1)
    q1_i = get_i(q1)
    q1_j = get_j(q1)
    q1_k = get_k(q1)

    # rr', xx', yy', and zz'
    r_base = torch.mul(q0, q1)
    # (rr' - xx' - yy' - zz')
    r = get_r(r_base) - get_i(r_base) - get_j(r_base) - get_k(r_base)

    # rx', xr', yz', and zy'
    i_base = torch.mul(q0, torch.cat([q1_i, q1_r, q1_k, q1_j], dim=1))
    # (rx' + xr' + yz' - zy')
    i = get_r(i_base) + get_i(i_base) + get_j(i_base) - get_k(i_base)

    # ry', xz', yr', and zx'
    j_base = torch.mul(q0, torch.cat([q1_j, q1_k, q1_r, q1_i], dim=1))
    # (ry' - xz' + yr' + zx')
    j = get_r(j_base) - get_i(j_base) + get_j(j_base) + get_k(j_base)

    # rz', xy', yx', and zr'
    k_base = torch.mul(q0, torch.cat([q1_k, q1_j, q1_i, q1_r], dim=1))
    # (rz' + xy' - yx' + zr')
    k = get_r(k_base) + get_i(k_base) - get_j(k_base) + get_k(k_base)

    return torch.cat([r, i, j, k], dim=1)
