# Quaternion Neural Networks

Welcome to the **Quaternion Neural Networks** library for PyTorch!

This project provides a robust, high-performance implementation of Quaternion Neural Networks (QNNs) in PyTorch. It is heavily inspired by and built upon the excellent [Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks) by Orkis Research, but extends the original work with modern software engineering practices, full type hinting, strict code quality enforcement, and robust distributed training capabilities.

## What are Quaternion Neural Networks?

Standard artificial neural networks use real numbers for weights and activations. Quaternion Neural Networks use hypercomplex numbers (quaternions) to encode information. Quaternions consist of one real part and three imaginary parts ($x = r + xi + yj + zk$). 

Because of their internal structure, quaternions are naturally suited to represent and process multi-dimensional inputs (such as 3D coordinates, RGB images, and multi-channel audio), capturing internal dependencies between features more effectively than standard dense layers, all while using up to 4x fewer parameters.

## Core Features

- **Standard Layers**: Full PyTorch implementations of Quaternion Convolution (`QuaternionConv`), Linear (`QuaternionLinear`), and Initialization.
- **Normalization**: Supports `QuaternionBatchNorm2d` and `QuaternionGroupNorm2d`.
- **Distributed Training (DDP)**: Offers robust `Synchronized_Quaternion_BatchNorm2d` implementations designed to aggregate statistics properly across multiple GPUs or machines during distributed training.
- **Modern Packaging**: Built using `uv` for lightning-fast dependency management.
- **High Quality**: Fully typed, tested, and linted using modern standard tools (`ruff`, `mypy`, `pytest`).

## Getting Started

Check out the [Getting Started](getting-started.md) guide for installation instructions, or dive straight into the [API Reference](api-reference.md).
