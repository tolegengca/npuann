# Artificial Neural Network (ANN) Platform
# NPU-ANN

Group Team:
1. Tolegen Aiteni
2. Erniyaz Ashuov
3. Daulet Akhmetbekov

---

## Overview

This project implements a modular, extensible Artificial Neural Network platform following Object-Oriented Programming principles. Every component — forward propagation, gradient computation, and parameter updates — is written explicitly without automatic differentiation.

---

## Features

| Component | Details |
|---|---|
| **Linear Layer** | Fully connected with He initialization, forward + backward |
| **Conv1D** | 1D convolution, valid & same padding, configurable stride |
| **Conv2D** | 2D convolution, valid & same padding, square or rectangular kernels |
| **ReLU** | Piecewise-linear activation, dead-neuron safe |
| **Sigmoid** | Saturating activation with overflow clipping |
| **Tanh** | Zero-centered saturating activation |
| **Softmax** | Numerically stable output activation |
| **CrossEntropyLoss** | Fused softmax + CE with stable log computation |
| **MSELoss** | Mean Squared Error for regression and gradient checks |
| **SGD + Momentum** | Velocity-based optimizer with per-parameter state |
| **Sequential** | Layer container with chained forward/backward |
| **Flatten** | Bridges convolutional and dense layers |
| **Gradient Check** | Numerical verification of backpropagation |

---

## Project Structure

```
ANN_Platform.ipynb          # Main Jupyter Notebook (all code + experiments)
README.md                   # This file
ANN_Platform_Report.docx    # Full technical report
ann_results.png             # Training plots (generated at runtime)
conv1d_filters.png          # Learned Conv1D filter visualization (generated at runtime)
```

---

## Requirements

The notebook uses only standard Python scientific libraries:

```
numpy
matplotlib
scikit-learn
```

No GPU or specialized hardware is required. Runs on Google Colab free tier.

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `ANN_Platform.ipynb`
3. Go to **Runtime → Run all** (or press `Ctrl+F9`)
4. Wait for MNIST to download (~1 min first run), then all experiments run automatically.

### Option 2: Local Jupyter

```bash
pip install numpy matplotlib scikit-learn jupyter
jupyter notebook ANN_Platform.ipynb
```

---

## Architecture

All layers inherit from a common `Layer` base class:

```python
class Layer:
    def forward(self, x):   raise NotImplementedError
    def backward(self, grad): raise NotImplementedError
    def get_params(self):   return []   # trainable (param, grad) pairs
```

Models are composed with `Sequential`:

```python
model = Sequential([
    Linear(784, 256), ReLU(),
    Linear(256, 128), ReLU(),
    Linear(128, 10)
])
```

---

## Layer Details

### Linear Layer

```
Forward:  out = X @ W + b
Backward: dL/dW = X.T @ grad_out
          dL/db = sum(grad_out, axis=0)
          dL/dX = grad_out @ W.T
```

Weights initialized with **He Normal**: `scale = sqrt(2 / fan_in)`

---

### Conv1D

```
Input:  (batch, length, in_channels)
Output: (batch, out_length, out_channels)
out_length (valid) = (length - kernel_size) // stride + 1
out_length (same)  = length
```

Backward pass uses `np.einsum` to accumulate gradients for W, b, and input.

---

### Conv2D

```
Input:  (batch, H, W, in_channels)
Output: (batch, out_H, out_W, out_channels)
out_H = (H - kH) // stride_H + 1   (valid)
out_W = (W - kW) // stride_W + 1   (valid)
```

Accepts `kernel_size` and `stride` as `int` or `(int, int)` tuple.

---

### Activation Functions

| Function | Forward | Gradient |
|---|---|---|
| ReLU | max(0, x) | 1 if x > 0, else 0 |
| Sigmoid | 1 / (1 + exp(-x)) | σ(x) · (1 − σ(x)) |
| Tanh | tanh(x) | 1 − tanh(x)² |
| Softmax | exp(xᵢ) / Σexp(xⱼ) | Identity (fused with CE) |

---

## Experiments

### A: MLP Baseline (MNIST)

```python
model = Sequential([
    Linear(784, 256), ReLU(),
    Linear(256, 128), ReLU(),
    Linear(128, 10)
])
# Expected: ~94-96% test accuracy in 30 epochs
```

### B: Activation Comparison

Same architecture (784→128→64→10) trained with ReLU, Sigmoid, and Tanh.

**Expected ranking:** ReLU > Tanh > Sigmoid

- **ReLU**: fastest convergence, ~93-95% acc
- **Tanh**: moderate convergence, ~91-93% acc
- **Sigmoid**: slowest due to gradient vanishing, ~87-91% acc

### C: CNN on MNIST

```python
model = Sequential([
    Conv2D(1, 8, kernel_size=3, padding='valid'),
    ReLU(),
    Flatten(),
    Linear(26*26*8, 64), ReLU(),
    Linear(64, 10)
])
# 2,000 samples used (conv2d is slow in pure NumPy)
# Expected: ~85-90% test accuracy in 15 epochs
```

### D: Conv1D on Synthetic Signals

Binary classification of sine vs. cosine waveforms (length 32, with noise).

```python
model = Sequential([
    Conv1D(1, 8, kernel_size=5, padding='valid'),
    ReLU(), Flatten(),
    Linear(28*8, 32), ReLU(),
    Linear(32, 2)
])
# Expected: >95% accuracy
```

---

## Gradient Verification

The notebook includes a numerical gradient check using central differences:

```
numerical_grad[i] = (L(x + eps*eᵢ) - L(x - eps*eᵢ)) / (2 * eps)
```

Compared to analytical backpropagation. Typical relative error: ~1e-6 (threshold: 1e-4).

---

## Unit Tests

Each layer is verified with shape assertions and value checks:

```
=== Test 1: Linear Layer ===
  Forward output shape: (4, 5)  ✓
  Backward grad shape:  (4, 8)  ✓

=== Test 2: ReLU Activation ===
  ReLU forward:  [0. 0. 0. 1. 2.]  ✓
  ReLU backward: [0. 0. 0. 1. 1.]  ✓

=== Test 3: Sigmoid Activation ===
  Sigmoid(0) = 0.5  ✓

=== Test 5: Conv1D Layer ===
  Conv1D (valid) output: (2, 8, 4)  ✓
  Conv1D (same)  output: (2, 10, 4) ✓

=== Test 6: Conv2D Layer ===
  Conv2D (valid) output: (2, 6, 6, 4)  ✓
  Conv2D (same)  output: (2, 8, 8, 4)  ✓
```

---

## Design Principles

- **No autograd**: Every derivative is hand-computed from first principles.
- **OOP**: Abstract base class + concrete implementations for clean extensibility.
- **NumPy only**: `np.einsum` for efficient batched tensor contractions.
- **Reproducibility**: `np.random.seed(42)` set at import for deterministic results.
- **Numerical stability**: Softmax and cross-entropy both use the log-sum-exp trick.

---

## Known Limitations

- **Conv speed**: The nested-loop Conv2D/Conv1D is O(H·W·kH·kW) per sample. For large images, `im2col` vectorization would be needed for practical training times.
- **No GPU**: Pure NumPy runs on CPU only.
- **No LR scheduling**: Fixed learning rate throughout training.
- **No BatchNorm / Dropout**: Not implemented (potential extension).

---

## Evaluation Criteria Coverage

| Criterion | Implementation |
|---|---|
| **Correctness** (30 pts) | All layers forward + backward; gradient check verified |
| **Efficiency** (20 pts) | `np.einsum` for vectorized tensor ops; He init for fast convergence |
| **Code Quality** (20 pts) | OOP design, docstrings, inline comments throughout |
| **Results & Analysis** (10 pts) | 6-panel training plots, activation comparison, summary table |
| **Documentation** (20 pts) | This README + full technical report |

---

## License

This project is submitted as coursework. Not for redistribution.
