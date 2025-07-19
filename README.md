# PyRat 🐀 - Rapid Artificial Training

> **PyRat (Rapid Artificial Training)** is a lightweight, pure‑Python neural network & experimentation toolkit focused on clarity, academic use, and fast iteration on classic classification problems (e.g. MNIST) without the overhead of large frameworks.

<p align="center">
  <!-- Badges -->
  <img alt="GPL-3.0 License" src="https://img.shields.io/badge/license-GPLv3-blue" />
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-orange" />
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/Klinker195/PyRat" />
  <img alt="Code Size" src="https://img.shields.io/github/languages/code-size/Klinker195/PyRat" />
</p>

---

## Table of Contents
1. [Why PyRat?](#why-pyrat)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Core Concepts](#core-concepts)
7. [API Overview](#api-overview)
8. [Hyperparameter Search](#hyperparameter-search)
9. [MNIST End‑to‑End Example](#mnist-end-to-end-example)
10. [Citation](#citation)


---

## Why PyRat?
PyRat is intentionally **minimal & transparent**. You can open any file and understand the full training pipeline (layers → forward pass → loss → backward pass → optimizer update → hyperparameter search) in minutes. Built for:

- **Students** learning backprop & optimization from first principles.  
- **Educators** needing a compact reference implementation.  

---

## Features
- Modular **`Layer` → `Model`** design (`DenseLayer` provided).
- Common **activations**: ReLU, Sigmoid, Tanh, Linear, Softmax.
- Multiple **weight initializers**: Xavier (normal/uniform), He (normal/uniform).
- **Losses**: MSE & Cross‑Entropy (with derivatives).
- **Optimizers**: RPROP (sign adaptive) & Adam (moment adaptive).
- Configurable **training loop**: batching, shuffling, accuracy tracking, early stopping (`patience`, `min_delta`), verbosity.
- **Hyperparameter search**: Grid Search & Random Search with cross‑validation + parallel execution.
- **Scoring registry**: accuracy (maximize) & cross‑entropy (minimize).
- **MNIST utilities**: download/load (IDX gzip), normalization, one‑hot labels, train/val split.
- Pure **NumPy** core—no large framework dependency.

---

## Installation
> Replace the PyPI command once published.

### From Source
```bash
git clone https://github.com/<your-org>/pyrat.git
cd pyrat
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### Optional Extras
```bash
pip install matplotlib seaborn joblib scikit-learn
```

---

## Quick Start
```python
from pyrat.layers import DenseLayer
from pyrat.models import Model
from pyrat.datasets import MNIST, train_val_split

# Load & prepare MNIST
train = MNIST(train=True)              # 60,000 samples
X = train.images.reshape(len(train), -1)  # flatten 28x28 → 784
y = train.labels                       # one-hot shape: (N, 10)

# Train/validation split
X_train, X_val, y_train, y_val = train_val_split(X, y, val_size=0.2, random_state=42)

# Build model
model = Model(loss_fn="cross-entropy", optimizer="adam")
model.add(DenseLayer(784, 128, activation_fn='relu', weight_init_fn='he'))
model.add(DenseLayer(128, 10, activation_fn='softmax'))

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=64,
    validation_data=(X_val, y_val),
    patience=5,
    min_delta=1e-5,
    verbose=1,
    random_state=42
)

preds = model.predict(X_val)
print('Validation predictions shape:', preds.shape)
```

---

## Project Structure
```
pyrat/
  datasets.py      # MNIST loader & train/val split
  functions.py     # Activations, weight inits, losses, scoring
  layers.py        # Layer base + DenseLayer
  models.py        # Model class (training loop, early stopping)
  optimizers.py    # RPROP & Adam
  tuning.py        # Grid & Random search with CV
  main.py          # Example script (MNIST experiment)
```

---

## Core Concepts

### Layers
A `Layer` defines `forward(x)` and `backward(grad_out)` plus shape metadata. `DenseLayer` performs `z = xW + b` and applies an activation; intermediates are cached for gradients.

### Activations
Declared in a registry (e.g. `'relu'`, `'tanh'`, `'softmax'`) each with a derivative. Add new ones by registering a pair (function, derivative).

### Weight Initialization
Xavier & He (normal/uniform variants) reduce vanishing/exploding gradients. Chosen via `weight_init_fn`.

### Model
Holds an ordered list of layers, a loss + derivative, and optimizer instance. Provides:
- `add(layer)`
- `forward(X)`
- `backward(y_true)`
- `fit(...)`
- `predict(X)`

### Losses
- **MSE** for regression or analysis.
- **Cross‑Entropy** for classification (expects probabilities + one‑hot labels).

### Optimizers
- **RPROP**: adjusts per‑parameter step sizes from gradient sign changes.
- **Adam**: adaptive moments with bias correction.

### Training Loop & Early Stopping
Each epoch: (shuffle) → batch iteration (forward/backward/update) → metric tracking → optional validation → early stopping check on `patience` & `min_delta`.

### Scoring & Metrics
Scoring functions define comparison direction and initial sentinel. Built‑in: `accuracy`, `cross-entropy`.

---

## API Overview

### `DenseLayer`
```python
DenseLayer(
    input_size: int,
    output_size: int,
    activation_fn: str = 'relu',
    weight_init_fn: str = 'xavier'
)
```

### `Model`
```python
Model(
    layers=None,
    loss_fn='cross-entropy',    # or 'mse'
    optimizer='rprop',          # or 'adam'
    opt_params=None             # dict of optimizer hyperparameters
)
```

Key `fit` args: `epochs`, `batch_size`, `validation_data`, `shuffle`, `patience`, `min_delta`, `verbose`, `random_state`.

### Optimizers
```python
RPROP(eta_plus=1.2, eta_minus=0.5, delta_min=1e-6, delta_max=50)
Adam(learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

### Search Utilities
```python
grid_search_cv(model_class, param_grid, X, y,
               validation_data=None, cv=3,
               scoring='accuracy', n_jobs=1,
               verbose=1, random_state=None)

random_search_cv(model_class, param_grid, n_iter, X, y,
                 validation_data=None, cv=3,
                 scoring='accuracy', n_jobs=1,
                 verbose=1, random_state=None)
```

Return dict includes: `best_score`, `best_params`, `best_model`, `results` (per config stats), histories, `search_time`.

---

## Hyperparameter Search

A parameter grid maps names → lists. Supported: model params (`loss_fn`, `optimizer`), optimizer hyperparameters, training loop values (`epochs`, `batch_size`, `shuffle`, `patience`), and `layers_config` (list of layer lists).  
- **Grid Search** enumerates full Cartesian product.  
- **Random Search** samples `n_iter` combinations uniformly.

### Example
```python
param_grid = {
  "loss_fn": ["cross-entropy"],
  "optimizer": ["rprop"],
  "eta_plus": [1.1, 1.2, 1.5],
  "eta_minus": [0.3, 0.4, 0.7],
  "delta_min": [1e-6],
  "delta_max": [50, 75],
  "layers_config": [[
      DenseLayer(784, 16, activation_fn='sigmoid'),
      DenseLayer(16, 10, activation_fn='softmax')
  ]],
  "epochs": [100],
  "batch_size": [60000],   # full batch
  "shuffle": [True]
}
```

---

## MNIST End‑to‑End Example
```python
from pyrat.datasets import MNIST, train_val_split
from pyrat.layers import DenseLayer
from pyrat.models import Model
from pyrat.tuning import random_search_cv

mnist_train = MNIST(train=True)
mnist_test  = MNIST(train=False)

X_full = mnist_train.images.reshape(mnist_train.num_samples, -1)
y_full = mnist_train.labels

X_train, X_val, y_train, y_val = train_val_split(X_full, y_full, val_size=0.2, random_state=42)

param_grid = {
    "loss_fn": ["cross-entropy"],
    "optimizer": ["adam", "rprop"],
    "learning_rate": [1e-3, 5e-4],
    "eta_plus": [1.2],
    "layers_config": [
        [DenseLayer(784, 128, 'relu'), DenseLayer(128, 10, 'softmax')],
        [DenseLayer(784, 64, 'tanh'), DenseLayer(64, 10, 'softmax')]
    ],
    "epochs": [30],
    "batch_size": [128],
    "shuffle": [True]
}

results = random_search_cv(
    model_class=Model,
    param_grid=param_grid,
    n_iter=4,
    X=X_train, y=y_train,
    validation_data=(X_val, y_val),
    cv=3,
    scoring="accuracy",
    verbose=1,
    n_jobs=4,
    random_state=42
)

best_model = results["best_model"]
X_test = mnist_test.images.reshape(mnist_test.num_samples, -1)
y_test = mnist_test.labels
pred_test = best_model.predict(X_test)
```

---

## Citation
```bibtex
@software{pyrat2025,
  title  = {PyRat: Rapid Artificial Training Library},
  author = {Gianluca Viscardi, Mattia Rossi},
  year   = {2025},
  url    = {https://github.com/Klinker195/PyRat}
}
```

---

**Happy Experimenting!** 🐀
