import pytest
import torch
from torch.autograd import gradcheck

from quaternion_neural_networks.quaternion_conv import QuatConv1d, QuatConv2d, QuatConv3d
from tests.check_robustness import check_gradient_and_overfit


def test_quaternion_conv1d():
    """
    Tests the 1D Quaternion Convolution layer.
    Ensures that the output shape is mathematically correct based on channel division,
    and runs a concrete robustness check (gradient sanity, NaNs, and overfitting).
    """
    layer = QuatConv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(2, 8, 32)
    out = layer(x)
    assert out.shape == (2, 16, 32), "QuatConv1d shape mismatch"
    check_gradient_and_overfit(layer, x)

def test_quaternion_conv2d():
    """
    Tests the 2D Quaternion Convolution layer.
    Ensures that spatial dimensions and quaternion channels are properly maintained,
    and validates gradient integrity and optimization viability.
    """
    layer = QuatConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(2, 8, 32, 32)
    out = layer(x)
    assert out.shape == (2, 16, 32, 32), "QuatConv2d shape mismatch"
    check_gradient_and_overfit(layer, x)

def test_quaternion_conv3d():
    """
    Tests the 3D Quaternion Convolution layer (Volumetric).
    Ensures that depth, height, width, and hypercomplex channels are preserved,
    and tests backward pass gradient health.
    """
    layer = QuatConv3d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(2, 8, 16, 16, 16)
    out = layer(x)
    assert out.shape == (2, 16, 16, 16, 16), "QuatConv3d shape mismatch"
    check_gradient_and_overfit(layer, x)

def test_quaternion_conv_divisibility():
    """
    Validates that the convolution layers strictly enforce that the number of
    input channels is divisible by 4 (the dimensions of a quaternion).
    """
    with pytest.raises(ValueError, match="in_channels must be divisible by 4"):
        QuatConv2d(in_channels=3, out_channels=16, kernel_size=3)

@pytest.mark.parametrize(("in_channels", "out_channels"), [(4, 8), (8, 4)])
def test_quaternion_conv2d_gradcheck(in_channels, out_channels):
    """
    Verifies that the analytical gradients for QuatConv2d exactly match numerical
    approximations using PyTorch's gradcheck. This proves that the composite forward
    pass is mathematically correct and generates sound analytical gradients.
    """
    # Use double precision for gradcheck stability
    layer = QuatConv2d(in_channels, out_channels, kernel_size=3, padding=1).double()
    x = torch.randn(2, in_channels, 5, 5, dtype=torch.double, requires_grad=True)

    # A functional wrapper that gradcheck can perturb inputs/weights against
    def func(inputs, r, i, j, k, bias):
        layer.r_weight.data = r
        layer.i_weight.data = i
        layer.j_weight.data = j
        layer.k_weight.data = k
        if bias is not None:
            layer.bias.data = bias
        return layer(inputs)

    # Test numerical vs analytical gradient
    test = gradcheck(func, (x, layer.r_weight, layer.i_weight, layer.j_weight, layer.k_weight, layer.bias), eps=1e-6, atol=1e-4)
    assert test, "Gradcheck failed for QuaternionConv2d"
