import torch

from quaternion_neural_networks.quaternion_norm import QuaternionBatchNorm2d, QuaternionGroupNorm2d
from tests.check_robustness import check_gradient_and_overfit


def test_quaternion_batchnorm2d():
    """
    Tests the QuaternionBatchNorm2d layer.
    Ensures that the output shape matches the input and verifies that gradients
    are properly computed and capable of overfitting a small batch without instability.
    """
    layer = QuaternionBatchNorm2d(num_features=16)
    x = torch.randn(2, 16, 32, 32)
    out = layer(x)
    assert out.shape == (2, 16, 32, 32), "QuaternionBatchNorm2d shape mismatch"
    check_gradient_and_overfit(layer, x)

def test_quaternion_groupnorm2d():
    """
    Tests the QuaternionGroupNorm2d layer.
    Validates dimensional correctness and evaluates gradient health to ensure
    there are no exploding/vanishing gradients during optimization.
    """
    # 16 channels // 4 = 4 quaternion features. Groups per quat = 1 => num_groups = 4.
    layer = QuaternionGroupNorm2d(groups_per_quat=1, num_channels=16)
    x = torch.randn(2, 16, 32, 32)
    out = layer(x)
    assert out.shape == (2, 16, 32, 32), "QuaternionGroupNorm2d shape mismatch"
    check_gradient_and_overfit(layer, x)

def test_numeric_batchnorm():
    """
    Strict numerical validation of the QuaternionBatchNorm2d logic.
    Manually calculates the mathematically correct quaternion variance
    (pooled across components) and ensures that the PyTorch module's
    output, running statistics, and gradients exactly match the manual mathematical derivation.
    """
    a = torch.rand(2, 16, 32, 32, requires_grad=True)
    # Using momentum=1 so running_mean and running_var are entirely replaced by the first batch
    bn = QuaternionBatchNorm2d(16, momentum=1, eps=1e-5, affine=False)
    bn.train()

    # 1. Forward pass using the module
    single_a_out = bn(a)
    loss1 = single_a_out.sum()
    loss1.backward()
    single_a_grad = a.grad.clone()

    # Clear grad for manual calculation
    a.grad.zero_()

    # 2. Manual calculation
    N, C, H, W = a.shape
    multi_a_in_ = a.view(N, 4, C // 4, H, W)

    # Independent means
    mean = multi_a_in_.mean(dim=[0, 3, 4], keepdim=True)

    # Variance pooled across the 4 components, calculated wrt independent means
    var = ((multi_a_in_ - mean) ** 2).mean(dim=[0, 1, 3, 4], keepdim=True)

    # Unbiased variance calculation for running_var validation
    n_elements = multi_a_in_.numel() / multi_a_in_.size(2)
    unbiased_var = var * n_elements / (n_elements - 1)

    # Normalization (using biased var + eps for the actual forward pass)
    manual_out = (multi_a_in_ - mean) * torch.rsqrt(var + 1e-5)
    manual_out = manual_out.view(N, C, H, W)

    loss2 = manual_out.sum()
    loss2.backward()
    manual_grad = a.grad.clone()

    # 3. Assertions using torch.testing.assert_close
    torch.testing.assert_close(single_a_out, manual_out, atol=1e-5, rtol=1e-4)

    # Check running statistics
    torch.testing.assert_close(bn.running_mean, mean, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(bn.running_var, unbiased_var, atol=1e-5, rtol=1e-4)

    # Check gradients
    torch.testing.assert_close(single_a_grad, manual_grad, atol=1e-5, rtol=1e-4)
