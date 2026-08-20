# File   : batchnorm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/primary/lib/nn/modules/batchnorm.py


# Distributed under MIT License.

# https://github.com/pytorch/pytorch/issues/1051


import collections
import contextlib

import torch

from quaternion_neural_networks.quaternion_norm import QuaternionBatchNorm2d

try:
    from torch.nn.parallel._functions import Broadcast, ReduceAddCoalesced
except ImportError:
    ReduceAddCoalesced = Broadcast = None  # type: ignore

try:
    from jactorch.parallel.comm import SyncPrimary
    from jactorch.parallel.data_parallel import JacDataParallel as DataParallelWithCallback
except ImportError:
    from quaternion_neural_networks.ddp.comm import SyncPrimary
    from quaternion_neural_networks.ddp.replicate import DataParallelWithCallback


__all__ = [
    "Synchronized_Quaternion_BatchNorm1d",
    "Synchronized_Quaternion_BatchNorm2d",
    "Synchronized_Quaternion_BatchNorm3d",
    "convert_model",
    "patch_sync_batchnorm",
]


def _sum_ft_for_var(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=[0, 1, -1], keepdim=True)


def _sum_ft_for_mean(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=[0, -1], keepdim=True)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple("_ChildMessage", ["sum", "ssum", "mean_n", "var_n"])
_PrimaryMessage = collections.namedtuple("_PrimaryMessage", ["sum", "inv_std"])


class _Synchronized_Quaternion_BatchNorm(QuaternionBatchNorm2d):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.001, affine: bool = True):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self._sync_primary = SyncPrimary(self._data_parallel_primary)
        self._is_parallel = False
        self._parallel_id = None
        self._worker_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return super().forward(input)

        # Resize the input to (B, 4, C, -1).
        N, C, H, W = input.shape
        input = input.view([N, 4, C // 4, -1])

        # Compute the sum and square-sum.
        mean_n = input.size(0) * input.size(-1)
        var_n = input.size(0) * input.size(1) * input.size(-1)

        input_sum = _sum_ft_for_mean(input)
        input_ssum = _sum_ft_for_var(input**2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_primary.run_primary(
                _ChildMessage(input_sum, input_ssum, mean_n, var_n)
            )
        else:
            mean, inv_std = self._worker_pipe.run_worker(
                _ChildMessage(input_sum, input_ssum, mean_n, var_n)
            )

        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = ((input - mean) * inv_std).view([N, C, H, W]) * self.weight + self.bias
        else:
            output = ((input - mean) * inv_std).view([N, C, H, W])

        # Reshape it.
        return output

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means primary device.
        if self._parallel_id == 0:
            ctx.sync_primary = self._sync_primary
        else:
            self._worker_pipe = ctx.sync_primary.register_worker(copy_id)

    def _data_parallel_primary(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        mean_n = sum([i[1].mean_n for i in intermediates])
        var_n = sum([i[1].var_n for i in intermediates])

        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)

        mean, inv_std = self._compute_mean_std(sum_, ssum, mean_n, var_n)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _PrimaryMessage(*broadcasted[i * 2 : i * 2 + 2])))

        return outputs

    def _add_weighted(self, dest, delta, alpha=1, beta=1, bias=0):
        """return *dest* by `dest := dest*alpha + delta*beta + bias`"""
        return dest * alpha + delta * beta + beta

    def _compute_mean_std(self, sum_, ssum, mean_n_size, var_n_size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the primary device."""
        assert mean_n_size > 1, (
            "BatchNorm computes unbiased standard-deviation, which requires size > 1."
        )
        assert var_n_size > 1, (
            "BatchNorm computes unbiased standard-deviation, which requires size > 1."
        )

        mean = sum_ / mean_n_size

        correction = (sum_**2 / mean_n_size).sum(dim=[0, 1, -1], keepdim=True)
        sumvar = ssum - correction

        unbias_var = sumvar / (var_n_size - 1)
        bias_var = sumvar / var_n_size

        if hasattr(torch, "no_grad"):
            with torch.no_grad():
                self.running_mean.copy_(
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean.data.view(self.running_mean.shape)
                )
                self.running_var.copy_(
                    (1 - self.momentum) * self.running_var
                    + self.momentum * unbias_var.data.view(self.running_var.shape)
                )
        else:
            self.running_mean.copy_(
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.data.view(self.running_mean.shape)
            )
            self.running_var.copy_(
                (1 - self.momentum) * self.running_var
                + self.momentum * unbias_var.data.view(self.running_var.shape)
            )

        return mean, bias_var.clamp(self.eps) ** -0.5


class Synchronized_Quaternion_BatchNorm1d(_Synchronized_Quaternion_BatchNorm):
    r"""Applies Synchronized Batch Normalization over a 2d or 3d input that is seen as a
    mini-batch.

    $$
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

        $$
    This module differs from the built-in PyTorch BatchNorm1d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal BatchNorm

    Args:
        num_features (int): num_features from an expected input of size
            `batch_size x num_features [x width]`
        eps (float): a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum (float): the value used for the running_mean and running_var
            computation. Default: 0.1
        affine (bool): a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: $(N, C)$ or $(N, C, L)$
        - Output: $(N, C)$ or $(N, C, L)$ (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm1d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")
        super()._check_input_dim(input)


class Synchronized_Quaternion_BatchNorm2d(_Synchronized_Quaternion_BatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    $$
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

        $$
    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features (int): num_features from an expected input of
            size batch_size x num_features x height x width
        eps (float): a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum (float): the value used for the running_mean and running_var
            computation. Default: 0.1
        affine (bool): a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: $(N, C, H, W)$
        - Output: $(N, C, H, W)$ (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")
        super()._check_input_dim(input)


class Synchronized_Quaternion_BatchNorm3d(_Synchronized_Quaternion_BatchNorm):
    r"""Applies Batch Normalization over a 5d input that is seen as a mini-batch
    of 4d inputs

    $$
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

        $$
    This module differs from the built-in PyTorch BatchNorm3d as the mean and
    standard-deviation are reduced across all devices during training.

    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.

    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).

    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric BatchNorm
    or Spatio-temporal BatchNorm

    Args:
        num_features (int): num_features from an expected input of
            size batch_size x num_features x depth x height x width
        eps (float): a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum (float): the value used for the running_mean and running_var
            computation. Default: 0.1
        affine (bool): a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: $(N, C, D, H, W)$
        - Output: $(N, C, D, H, W)$ (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm3d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
        super()._check_input_dim(input)


@contextlib.contextmanager
def patch_sync_batchnorm():
    import torch.nn as nn

    backup = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d

    nn.BatchNorm1d = Synchronized_Quaternion_BatchNorm1d
    nn.BatchNorm2d = Synchronized_Quaternion_BatchNorm2d
    nn.BatchNorm3d = Synchronized_Quaternion_BatchNorm3d

    yield

    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d = backup


def convert_model(module):
    """Traverse the input module and its child recursively
       and replace all instance of torch.nn.modules.batchnorm.BatchNorm*N*d
       to SynchronizedBatchNorm*N*d
    Args:
        module: the input module needs to be convert to SyncBN model
    Examples:
        >>> import torch.nn as nn
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = nn.DataParallel(m)
        >>> # after convert, m is using SyncBN
        >>> m = convert_model(m)
    """
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model(mod)
        mod = DataParallelWithCallback(mod, device_ids=module.device_ids)
        return mod

    mod = module
    for pth_module, sync_module in zip(
        [QuaternionBatchNorm2d, QuaternionBatchNorm2d, QuaternionBatchNorm2d],
        [
            Synchronized_Quaternion_BatchNorm1d,
            Synchronized_Quaternion_BatchNorm2d,
            Synchronized_Quaternion_BatchNorm3d,
        ],
        strict=False,
    ):
        if isinstance(module, pth_module):
            mod = sync_module(module.num_features, module.eps, module.momentum, module.affine)
            mod.running_mean = module.running_mean
            mod.running_var = module.running_var
            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_model(child))

    return mod
