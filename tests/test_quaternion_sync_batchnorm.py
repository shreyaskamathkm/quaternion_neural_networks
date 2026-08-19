import threading
from unittest.mock import patch

import torch

from quaternion_neural_networks.quaternion_sync_batchnorm import (
    Synchronized_Quaternion_BatchNorm2d,
)
from tests.check_robustness import check_gradient_and_overfit


def test_quaternion_sync_batchnorm2d():
    """
    Tests the Synchronized_Quaternion_BatchNorm2d layer in single-device fallback mode.
    Ensures that the output shape matches the input and verifies that gradients
    are properly computed and capable of overfitting a small batch without instability.
    """
    layer = Synchronized_Quaternion_BatchNorm2d(num_features=16)
    x = torch.randn(2, 16, 32, 32)
    out = layer(x)
    assert out.shape == (2, 16, 32, 32), "Synchronized_Quaternion_BatchNorm2d shape mismatch"
    check_gradient_and_overfit(layer, x)


def test_quaternion_sync_batchnorm2d_multi_thread():
    """
    Rigorously tests the Synchronized_Quaternion_BatchNorm2d layer in a simulated
    DataParallel environment to prove that the cross-device SyncPrimary
    aggregation computes global mean and variance correctly.
    """

    class MockContext:
        pass

    class MockReduceAddCoalesced:
        @staticmethod
        def apply(destination, num_inputs, *tensors):
            # tensors contains `num_inputs` tensors from each device sequentially.
            # e.g., for 2 devices and num_inputs=2: [dev1_t1, dev1_t2, dev2_t1, dev2_t2]
            n_devices = len(tensors) // num_inputs
            results = []
            for t_idx in range(num_inputs):
                total = sum(tensors[d_idx * num_inputs + t_idx] for d_idx in range(n_devices))
                results.append(total)
            return tuple(results)

    class MockBroadcast:
        @staticmethod
        def apply(target_gpus, *tensors):
            # Returns a tuple of broadcasted tensors
            res = []
            for _ in target_gpus:
                res.extend(tensors)
            return tuple(res)

    # Mock get_device to always return 0 (simulating CPU as a single GPU for broadcasting)
    with (
        patch("torch.Tensor.get_device", return_value=0),
        patch(
            "quaternion_neural_networks.quaternion_sync_batchnorm.ReduceAddCoalesced",
            MockReduceAddCoalesced,
        ),
        patch("quaternion_neural_networks.quaternion_sync_batchnorm.Broadcast", MockBroadcast),
    ):
        layer_primary = Synchronized_Quaternion_BatchNorm2d(num_features=16)
        layer_worker = Synchronized_Quaternion_BatchNorm2d(num_features=16)

        # Synchronize their weights and buffers
        layer_worker.load_state_dict(layer_primary.state_dict())

        ctx = MockContext()
        layer_primary.__data_parallel_replicate__(ctx, 0)  # Primary
        layer_worker.__data_parallel_replicate__(ctx, 1)  # Worker

        layer_primary.train()
        layer_worker.train()

        # We will simulate data from 2 devices
        # True global batch has mean 5, and var 25 (std = 5)
        x_primary = torch.zeros(2, 16, 2, 2)
        x_worker = torch.ones(2, 16, 2, 2) * 10.0

        primary_out = [None]
        worker_out = [None]

        def run_primary():
            primary_out[0] = layer_primary(x_primary)

        def run_worker():
            worker_out[0] = layer_worker(x_worker)

        t2 = threading.Thread(target=run_worker)
        t1 = threading.Thread(target=run_primary)

        t2.start()
        t1.start()

        t1.join()
        t2.join()

        # Primary data was 0. Global mean is 5. Global std is 5.
        # Normalized output should be (0 - 5) / 5 = -1
        # Allow small numerical tolerance
        assert torch.allclose(primary_out[0], torch.ones_like(primary_out[0]) * -1.0, atol=1e-3), (
            "Primary sync output incorrect, failed global reduction"
        )

        # Worker data was 10. Global mean is 5. Global std is 5.
        # Normalized output should be (10 - 5) / 5 = +1
        assert torch.allclose(worker_out[0], torch.ones_like(worker_out[0]) * 1.0, atol=1e-3), (
            "Worker sync output incorrect, failed global reduction"
        )

        # Also check running mean updated correctly.
        # Momentum = 0.001 (default). 0 * 0.999 + 5.0 * 0.001 = 0.005
        assert torch.allclose(
            layer_primary.running_mean,
            torch.ones_like(layer_primary.running_mean) * 0.005,
            atol=1e-3,
        ), "Running mean not updated correctly via SyncPrimary"
