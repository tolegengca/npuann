# npuann — ANN Platform

**Artificial Neural Network platform implemented from scratch using NumPy — no PyTorch, TensorFlow, or autograd.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Components](#components)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Building a model](#building-a-model)
  - [Training](#training)
  - [Running experiments](#running-experiments)
- [Experiments](#experiments)
- [Testing & Gradient Check](#testing--gradient-check)
- [Notebook Structure](#notebook-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project is a self-contained deep-learning framework built on top of **NumPy only**.  
It provides every building block needed to construct, train, and evaluate neural networks — including
fully connected layers, 1-D and 2-D convolutional layers, activation functions, loss functions, and
an SGD optimiser — all with hand-written forward and backward passes.

The included Jupyter notebook (`ANN_Platform (1) (1).ipynb`) walks through the full implementation
and evaluates the models on a 10 000-sample subset of **MNIST**.

---

## Components

| Component | Class | Description |
|---|---|---|
| **Base layer** | `Layer` | Abstract base; all layers implement `forward()` / `backward()` |
| **Linear layer** | `Linear` | Fully connected layer, He initialisation |
| **1-D Conv** | `Conv1D` | Valid & same padding, configurable kernel/stride |
| **2-D Conv** | `Conv2D` | Valid & same padding, square or rectangular kernels |
| **Flatten** | `Flatten` | Reshapes spatial output to 1-D feature vector |
| **ReLU** | `ReLU` | `f(x) = max(0, x)` |
| **Sigmoid** | `Sigmoid` | `f(x) = 1 / (1 + e^-x)` |
| **Tanh** | `Tanh` | `f(x) = tanh(x)` |
| **Softmax** | `Softmax` | Numerically stable softmax |
| **Cross-entropy** | `CrossEntropyLoss` | Softmax + cross-entropy, includes accuracy helper |
| **MSE** | `MSELoss` | Mean squared error (useful for regression / gradient checks) |
| **Optimiser** | `SGD` | Stochastic Gradient Descent with momentum |
| **Sequential** | `Sequential` | Chains layers; handles forward / backward pass |

---

## Requirements

- Python 3.8+
- numpy
- matplotlib
- scikit-learn (for MNIST loading and train/test split)

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Getting Started

```bash
git clone https://github.com/tolegengca/npuann.git
cd npuann
jupyter notebook "ANN_Platform (1) (1).ipynb"
```

---

## Usage

### Building a model

```python
# Multi-layer perceptron
mlp = Sequential([
    Linear(784, 256), ReLU(),
    Linear(256, 128), ReLU(),
    Linear(128, 10)
])

# Small CNN
cnn = Sequential([
    Conv2D(1, 8, kernel_size=3, stride=1, padding='valid'),  # (N,28,28,1) → (N,26,26,8)
    ReLU(),
    Flatten(),
    Linear(26 * 26 * 8, 64),
    ReLU(),
    Linear(64, 10)
])
```

### Training

```python
optimizer = SGD(lr=0.05, momentum=0.9)
loss_fn   = CrossEntropyLoss()

history = train(
    model, optimizer, loss_fn,
    X_train, y_train, X_test, y_test,
    epochs=30, batch_size=128, print_every=5
)
```

The `train()` helper runs a mini-batch SGD loop and returns a history dict with
`train_loss`, `train_acc`, `test_loss`, and `test_acc` per epoch.

### Running experiments

Open the notebook and run all cells in order. Each section is self-contained and
prints progress as it trains.

---

## Experiments

| Experiment | Architecture | Notes |
|---|---|---|
| **A — MLP** | `784 → 256 → 128 → 10` | 30 epochs, batch 128, SGD lr=0.05 |
| **B — Activation comparison** | `784 → 128 → 64 → 10` × 3 | Compares ReLU vs Sigmoid vs Tanh |
| **C — Small CNN** | `Conv2D(1→8,3) → ReLU → Flatten → 64 → 10` | 2 000-sample subset (pure-NumPy conv is slow) |

Loss and accuracy curves are plotted for all experiments.

---

## Testing & Gradient Check

The notebook includes unit tests for every layer (forward shape, backward shape, and
correctness) and a **numerical gradient check** for the Linear layer using central
differences:

```
Max relative error < 1e-4  →  ✓ PASS
```

---

## Notebook Structure

| Section | Content |
|---|---|
| 1 | Imports & setup |
| 2 | Base `Layer` class |
| 3 | Activation functions (ReLU, Sigmoid, Tanh, Softmax) |
| 4 | Linear (fully connected) layer |
| 5 | Conv1D layer |
| 6 | Conv2D layer |
| 7 | Flatten layer |
| 8 | Loss functions (CrossEntropy, MSE) |
| 9 | SGD optimiser |
| 10 | Sequential model container |
| 11 | Unit tests — each layer |
| 12 | Gradient check (numerical vs analytical) |
| 13 | Load & prepare MNIST data |
| 14 | Training loop |
| 15 | Experiment A — pure MLP |
| 16 | Experiment B — activation comparison |
| 17 | Experiment C — small CNN |
| 18 | Loss & accuracy plots |
| 19 | Final summary table |
| 20 | Conv1D filter visualisation on synthetic 1-D signals |

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add my feature"`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Open a Pull Request.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
