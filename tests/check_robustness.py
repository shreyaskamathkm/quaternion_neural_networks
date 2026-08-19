import torch
import torch.nn.functional as F


def check_gradient_and_overfit(model, x, y_target=None):
    """
    Concrete test to ensure gradient stability, lack of NaNs, and ability to overfit a small batch.
    """
    model.train()

    # Generate random target if none provided
    with torch.no_grad():
        if y_target is None:
            out_shape = model(x).shape
            y_target = torch.randn(*out_shape, device=x.device)

    # 1. Forward and Backward Sanity (No NaNs, No Infs)
    x.requires_grad = True
    out = model(x)
    loss = F.mse_loss(out, y_target)
    loss.backward()

    # Check input gradients
    assert x.grad is not None, "Input gradient is None"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaNs"
    assert not torch.isinf(x.grad).any(), "Input gradient contains Infs"
    assert x.grad.abs().sum() > 0, "Input gradient is entirely zero"

    # Check parameter gradients
    has_params = False
    for name, param in model.named_parameters():
        has_params = True
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"Gradient of {name} contains NaNs"
            assert not torch.isinf(param.grad).any(), f"Gradient of {name} contains Infs"
            assert param.grad.abs().sum() > 0, f"Gradient of {name} is entirely zero"

    # 2. Optimization Sanity Check (Overfitting a single batch)
    if not has_params:
        return # Skip optimization if no trainable parameters (e.g. affine=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # We detach x for the optimization loop to focus on model parameters
    x_opt = x.detach()

    initial_loss = None
    final_loss = None
    for i in range(50):
        optimizer.zero_grad()
        out = model(x_opt)
        loss = F.mse_loss(out, y_target)
        if i == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss, f"Model failed to overfit: initial loss {initial_loss:.6f}, final loss {final_loss:.6f}"
