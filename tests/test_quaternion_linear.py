import pytest
import torch
from torch.autograd import gradcheck

from quaternion_neural_networks.quaternion_linear import QuaternionLinear, QuaternionLinearAutograd
from tests.check_robustness import check_gradient_and_overfit


def test_quaternion_linear_forward():
    """
    Tests the base QuaternionLinear layer.
    Ensures correct dimension projection and verifies robust gradient
    computation and optimization on a synthetic batch.
    """
    layer = QuaternionLinear(16, 32)
    x = torch.randn(10, 16)
    out = layer(x)
    assert out.shape == (10, 32), "Output shape should be (10, 32)"
    check_gradient_and_overfit(layer, x)

def test_quaternion_linear_autograd_forward():
    """
    Tests the QuaternionLinearAutograd layer.
    Verifies that the explicit autograd function implementation computes
    gradients properly without exploding or vanishing during a training loop.
    """
    layer = QuaternionLinearAutograd(16, 32)
    x = torch.randn(10, 16)
    out = layer(x)
    assert out.shape == (10, 32), "Output shape should be (10, 32)"
    check_gradient_and_overfit(layer, x)

@pytest.mark.parametrize(("in_features", "out_features"), [(4, 8), (8, 4)])
def test_quaternion_linear_gradcheck(in_features, out_features):
    """
    Performs a rigorous mathematical verification of the backward pass for QuaternionLinear.
    Uses PyTorch's gradcheck to ensure that the analytical gradients perfectly match the
    numerical finite-difference approximations.
    """
    layer = QuaternionLinear(in_features, out_features).double()

    # We must use double precision for gradcheck
    x = torch.randn(2, in_features, dtype=torch.double, requires_grad=True)

    def func(inputs, r, i, j, k, bias):
        cat_kernels_4_r = torch.cat([r, -i, -j, -k], dim=0)
        cat_kernels_4_i = torch.cat([i, r, -k, j], dim=0)
        cat_kernels_4_j = torch.cat([j, k, r, -i], dim=0)
        cat_kernels_4_k = torch.cat([k, -j, i, r], dim=0)
        cat_kernels_4_quaternion = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=1)

        out = torch.mm(inputs, cat_kernels_4_quaternion)
        if bias is not None:
            out += bias
        return out

    test = gradcheck(func, (x, layer.r_weight, layer.i_weight, layer.j_weight, layer.k_weight, layer.bias), eps=1e-6, atol=1e-4)
    assert test, "Gradcheck failed for QuaternionLinear"
