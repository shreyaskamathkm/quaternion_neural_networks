# Getting Started

Welcome to the **Quaternion Neural Networks** library! We've designed this library to feel as familiar as possible for PyTorch users. If you know how to build a standard neural network in PyTorch, you already know how to build a Quaternion Neural Network.

---

## 📦 Installation

To get started, simply install the library into your environment. You can install it via pip:

```bash
pip install quaternion_neural_networks
```

If you are developing or modifying the library itself, we recommend using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/shreyaskamathkm/quaternion_neural_networks.git
cd quaternion_neural_networks
make edit-install
```

---

## ⚡ Quick Start: Your First Quaternion Network

Building a quaternion network is exactly like building a standard real-valued neural network, with one important mathematical distinction: **your channel/feature dimensions must be divisible by 4**.

This is because each quaternion consists of 4 components (1 real part + 3 imaginary parts). Our layers expect to process inputs where the features are grouped into quaternions.

### Example: A Simple Quaternion CNN

Let's build a simple 2D convolutional network. Notice how we use `QuaternionConv2d`, `QuaternionBatchNorm2d`, and `QuaternionLinear` just as drop-in replacements for their standard `torch.nn` equivalents!

```python
import torch
import torch.nn as nn
from quaternion_neural_networks.quaternion_conv import QuaternionConv2d
from quaternion_neural_networks.quaternion_linear import QuaternionLinear
from quaternion_neural_networks.quaternion_norm import QuaternionBatchNorm2d

class QuaternionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 4 input channels (e.g., RGBA image or 3D point cloud + intensity) -> 16 output channels
        # Note: Both 4 and 16 are divisible by 4!
        self.conv1 = QuaternionConv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = QuaternionBatchNorm2d(num_features=16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 16 channels * 14 * 14 spatial dimensions = 3136 input features (divisible by 4)
        self.fc1 = QuaternionLinear(in_features=3136, out_features=64)
        
        # 64 features -> 10 output classes (Wait, 10 is NOT divisible by 4!)
        # If your final output classes aren't divisible by 4, you can transition 
        # back to standard PyTorch layers for the very last step:
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Convolutional Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Linear Block
        x = self.fc1(x)
        x = self.relu(x)
        
        # Standard Linear layer for classification
        x = self.fc2(x)
        return x

# Let's test it!
model = QuaternionCNN()

# Create dummy input: [Batch Size=8, Channels=4, Height=28, Width=28]
dummy_input = torch.randn(8, 4, 28, 28)

# Run the model
output = model(dummy_input)

print(f"Output shape: {output.shape}") 
# Expected: torch.Size([8, 10])
```

---

## 🛠️ Key Concepts

### 1. The "Divisible by 4" Rule
Because the core fundamental unit is a quaternion, the `in_channels` and `out_channels` (or `in_features` / `out_features`) of any Quaternion layer **must always be a multiple of 4**. 

If you have data with 3 channels (like standard RGB images), you can pad it with a zero channel, a grayscale channel, or use an initial standard `nn.Conv2d` layer to project the 3 channels into a dimension that is divisible by 4 before applying the Quaternion layers.

### 2. Distributed Training (DDP)
We provide a highly robust implementation of `Synchronized_Quaternion_BatchNorm2d`. Standard batch normalization can suffer in distributed training because it only normalizes across the local batch on each GPU. Our synchronized version properly aggregates statistics across all devices, maintaining the mathematical integrity of the quaternion normalization.

Check out the [API Reference](api-reference.md) for more detailed documentation on all available layers and initialization methods!
