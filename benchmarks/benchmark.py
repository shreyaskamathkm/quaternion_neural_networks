import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time

import torch
import torch.nn.functional as F

from benchmarks.naive_layers import NaiveQuaternionBatchNorm2d
from quaternion_neural_networks.quaternion_conv import QuatConv2d
from quaternion_neural_networks.quaternion_linear import QuaternionLinear


def benchmark_linear():
    print("--- Linear Benchmark ---")
    in_features = 1024
    out_features = 1024
    batch_size = 32

    # Original implementation using cat
    layer = QuaternionLinear(in_features, out_features).cuda()
    input_tensor = torch.randn(batch_size, in_features, device="cuda", requires_grad=True)

    # Warmup
    for _ in range(5):
        out = layer(input_tensor)
        out.sum().backward()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = layer(input_tensor)
        out.sum().backward()
    torch.cuda.synchronize()
    print(f"Original (Custom Autograd + Cat): {time.time() - start:.4f} seconds")

    # 16 Op Method
    r_w, i_w, j_w, k_w = layer.r_weight, layer.i_weight, layer.j_weight, layer.k_weight
    bias = layer.bias

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        r, i, j, k = torch.chunk(input_tensor, 4, dim=1)
        y_r = r.mm(r_w) - i.mm(i_w) - j.mm(j_w) - k.mm(k_w)
        y_i = r.mm(i_w) + i.mm(r_w) + j.mm(k_w) - k.mm(j_w)
        y_j = r.mm(j_w) - i.mm(k_w) + j.mm(r_w) + k.mm(i_w)
        y_k = r.mm(k_w) + i.mm(j_w) - j.mm(i_w) + k.mm(r_w)
        out2 = torch.cat([y_r, y_i, y_j, y_k], dim=1)
        if bias is not None:
            out2 += bias
        out2.sum().backward()
    torch.cuda.synchronize()
    print(f"16-Op Native MM: {time.time() - start:.4f} seconds")


def benchmark_conv():
    print("--- Conv Benchmark ---")
    in_channels = 64
    out_channels = 64
    k = 3
    batch_size = 16
    h, w = 32, 32

    layer = QuatConv2d(in_channels, out_channels, k, padding=1).cuda()
    input_tensor = torch.randn(batch_size, in_channels, h, w, device="cuda", requires_grad=True)

    # Warmup
    for _ in range(5):
        out = layer(input_tensor)
        out.sum().backward()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = layer(input_tensor)
        out.sum().backward()
    torch.cuda.synchronize()
    print(f"Original (Cat Weight): {time.time() - start:.4f} seconds")

    r_w, i_w, j_w, k_w = layer.r_weight, layer.i_weight, layer.j_weight, layer.k_weight
    bias = layer.bias

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        r, i, j, k = torch.chunk(input_tensor, 4, dim=1)
        y_r = (
            F.conv2d(r, r_w, padding=1)
            - F.conv2d(i, i_w, padding=1)
            - F.conv2d(j, j_w, padding=1)
            - F.conv2d(k, k_w, padding=1)
        )
        y_i = (
            F.conv2d(r, i_w, padding=1)
            + F.conv2d(i, r_w, padding=1)
            + F.conv2d(j, k_w, padding=1)
            - F.conv2d(k, j_w, padding=1)
        )
        y_j = (
            F.conv2d(r, j_w, padding=1)
            - F.conv2d(i, k_w, padding=1)
            + F.conv2d(j, r_w, padding=1)
            + F.conv2d(k, i_w, padding=1)
        )
        y_k = (
            F.conv2d(r, k_w, padding=1)
            + F.conv2d(i, j_w, padding=1)
            - F.conv2d(j, i_w, padding=1)
            + F.conv2d(k, r_w, padding=1)
        )
        out2 = torch.cat([y_r, y_i, y_j, y_k], dim=1)
        if bias is not None:
            out2 += bias.view(1, -1, 1, 1)
        out2.sum().backward()
    torch.cuda.synchronize()
    print(f"16-Op Native Conv: {time.time() - start:.4f} seconds")


def benchmark_batchnorm():
    print("--- BatchNorm Benchmark ---")
    channels = 64
    batch_size = 32
    h, w = 64, 64

    layer = NaiveQuaternionBatchNorm2d(channels).cuda()
    input_tensor = torch.randn(batch_size, channels, h, w, device="cuda", requires_grad=True)

    # Warmup
    for _ in range(5):
        out = layer(input_tensor)
        out.sum().backward()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = layer(input_tensor)
        out.sum().backward()
    torch.cuda.synchronize()
    print(f"Original Naive: {time.time() - start:.4f} seconds")

    opt_layer = torch.nn.BatchNorm2d(channels).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = opt_layer(input_tensor)
        out.sum().backward()
    torch.cuda.synchronize()
    print(f"Optimized Naive (Std BN): {time.time() - start:.4f} seconds")


if __name__ == "__main__":
    benchmark_linear()
    benchmark_conv()
    benchmark_batchnorm()
