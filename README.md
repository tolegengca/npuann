# npuann

**NPU-ANN** is a lightweight Artificial Neural Network framework focused on Convolutional Neural Networks (CNN), designed for deployment and acceleration on Neural Processing Units (NPUs).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Building a CNN Model](#building-a-cnn-model)
  - [Training](#training)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

NPU-ANN provides an easy-to-use interface for constructing, training, and deploying Convolutional Neural Networks. The framework is optimised to take advantage of Neural Processing Unit (NPU) hardware acceleration, enabling fast inference on edge devices while keeping a clean, developer-friendly API.

---

## Features

- **CNN Layer Support** – Convolution, pooling, batch normalisation, activation, and fully-connected layers out of the box.
- **NPU Acceleration** – Designed to offload computation to NPU hardware for faster inference.
- **Flexible Architecture** – Build custom network topologies with a modular, composable layer API.
- **Training & Evaluation** – Built-in training loop with support for common loss functions and optimisers.
- **Model Export** – Export trained models for deployment on target hardware.
- **Lightweight** – Minimal dependencies, suitable for embedded and edge environments.

---

## Architecture

```
Input → [Conv → BatchNorm → ReLU] × N → Pooling → Flatten → [FC → ReLU] × M → Softmax → Output
```

The framework follows a standard feed-forward pipeline. Each stage is represented as an independent, composable module so you can mix and match layers freely.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher (if using the Python interface)
- A compatible NPU SDK (optional, for hardware acceleration)
- `numpy` ≥ 1.21

### Installation

Clone the repository:

```bash
git clone https://github.com/tolegengca/npuann.git
cd npuann
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Building a CNN Model

```python
from npuann import Sequential
from npuann.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Softmax

model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding="same"),
    ReLU(),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding="same"),
    ReLU(),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=128),
    ReLU(),
    Dense(units=10),
    Softmax(),
])

model.summary()
```

### Training

```python
from npuann.optimizers import Adam
from npuann.losses import CrossEntropyLoss

model.compile(optimizer=Adam(lr=1e-3), loss=CrossEntropyLoss())
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
```

### Inference

```python
predictions = model.predict(x_test)
```

---

## Project Structure

```
npuann/
├── npuann/          # Core framework source code
│   ├── layers/      # Layer implementations (Conv, Pool, Dense, …)
│   ├── losses/      # Loss function implementations
│   ├── optimizers/  # Optimiser implementations
│   └── npu/         # NPU backend and hardware abstraction
├── examples/        # Example scripts and notebooks
├── tests/           # Unit and integration tests
├── LICENSE
└── README.md
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add my feature"`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Open a Pull Request.

Please make sure your code follows the existing style and includes relevant tests.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
