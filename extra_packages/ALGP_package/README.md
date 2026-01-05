# Active Learning and Gaussian Process (ALGP)

A Python package for Active Learning with Gaussian Process regression. ALGP intelligently selects the most informative training samples to build accurate GP models with minimal labeled data.

## Table of Contents

- [Active Learning and Gaussian Process (ALGP)](#active-learning-and-gaussian-process-algp)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Concepts](#key-concepts)
  - [Features](#features)
  - [Installation](#installation)
    - [From Source](#from-source)
    - [Requirements](#requirements)
  - [Quick Start](#quick-start)
    - [Toy Example](#toy-example)
  - [Usage Guide](#usage-guide)
    - [Basic Example](#basic-example)
    - [Data Preparation](#data-preparation)
    - [Configuration](#configuration)
  - [API Reference](#api-reference)
    - [ALGP Function](#algp-function)
    - [GPModel Class](#gpmodel-class)
  - [Active Learning Strategy](#active-learning-strategy)
  - [Project Structure](#project-structure)
  - [Requirements](#requirements-1)

## Overview

ALGP (Active Learning with Gaussian Process) implements an uncertainty-based active learning strategy for Gaussian Process regression. Instead of training on the entire dataset, ALGP iteratively selects the most informative samples based on prediction uncertainty, enabling efficient model training with reduced labeled data.

### Key Concepts

- **Active Learning**: Intelligently selects training samples to maximize model performance with minimal data
- **Gaussian Process**: Provides probabilistic predictions with uncertainty estimates
- **Uncertainty Sampling**: Selects samples with highest prediction uncertainty (standard deviation)

## Features

- 🎯 **Smart Sample Selection**: Automatically selects most informative training samples based on GP uncertainty
- 📊 **Multi-Output Support**: Handles multiple output targets with independent GP models
- ⚙️ **Configurable**: Flexible hyperparameter configuration via YAML files
- 📈 **Built-in Evaluation**: Comprehensive metrics for model performance assessment
- 🔧 **Scikit-learn Backend**: Leverages sklearn's GaussianProcessRegressor

## Installation

### From Source

```bash
# Conda virtual env
conda create --name ALGP-env python=3.10 -y
conda activate ALGP-env

# Clone the repository
cd /path/to/ALGP_package

# Install in editable mode
pip install -e .
```

### Requirements

The package requires:
- Python >= 3.7
- scikit-learn
- omegaconf
  
All dependencies will be installed automatically.

## Quick Start

### Toy Example

```python
from ALGP.ALGP_module import ALGP
from ALGP import AP
import numpy as np

# Prepare your data
x_train = np.random.rand(1000, 5)  # 1000 samples, 5 features
y_train = np.random.rand(1000, 4)  # 1000 samples, 4 outputs

# Train with active learning
model = ALGP(
    x_train=x_train,
    y_train=y_train,
    index_selection_target=0,  # Use first output for sample selection
    Budget_Training_Sample=50,  # Use only 50 samples
    NUM_INIT_SAMPLE=10,        # Start with 10 random samples
    TOP_K=1,                   # Select 1 sample per iteration
    gp_config_path=AP('configs/gp-l_bfgs_v01.yaml')
)

# Make predictions
x_test = np.random.rand(100, 5)
y_pred, y_std = model.predict(x_test, return_std=True)
```

## Usage Guide

### Basic Example

The package includes a complete example in [scripts/train_ALGP_example.py](scripts/train_ALGP_example.py):

```bash
python scripts/train_ALGP_example.py
```

This example demonstrates:
1. Loading and preprocessing data
2. Training an ALGP model with active learning
3. Making predictions with uncertainty estimates
4. Evaluating model performance

### Data Preparation

Your data should be formatted as NumPy arrays:

```python
# Features (X): 2D array of shape (n_samples, n_features)
x_train = np.array([[x1_1, x1_2, ...],
                    [x2_1, x2_2, ...],
                    ...])

# Targets (y): 2D array of shape (n_samples, n_outputs)
y_train = np.array([[y1_1, y1_2, ...],
                    [y2_1, y2_2, ...],
                    ...])
```

**Important**: Data should be standardized (zero mean, unit variance) for optimal GP performance.

### Configuration

GP hyperparameters are configured via YAML files in the [configs](configs) folder. Default configuration [configs/gp-l_bfgs_v01.yaml](configs/gp-l_bfgs_v01.yaml):

```yaml
length_scale: 1                   # Kernel length scale
nu: 2.5                           # Matern kernel smoothness parameter
noise_level: 1e-7                 # GP noise level
noise_level_bounds_lower: 1e-10   # Lower bound for the noise level
noise_level_bounds_upper: 1       # Upper bound for the noise level
alpha: 1e-7                       # Additive noise for numerical stability
gp_optimizer: 'fmin_l_bfgs_b'     # Hyperparameter optimizer
n_restarts_optimizer: 7           # Number of optimizer restarts
```

## API Reference

### ALGP Function

Main function for training an active learning Gaussian Process model.

```python
from ALGP.ALGP_module import ALGP

model = ALGP(
    x_train,                        # Training features (n_samples, n_features)
    y_train,                        # Training targets (n_samples, n_outputs)
    ActiveLearning_Target_Index=0,  # Output index for sample selection
    Budget_Training_Sample=50,      # Maximum training samples
    NUM_INIT_SAMPLE=10,             # Initial random samples
    TOP_K=1,                        # Samples to select per iteration
    gp_config_path=None             # Path to GP config YAML
)
```

**Parameters:**

- **x_train** (np.ndarray): Training features (x), shape `(n_samples, n_features)`
- **y_train** (np.ndarray): Training targets (y), shape `(n_samples, n_outputs)`
- **ActiveLearning_Target_Index** (int, default=0): Active learning loop target. Which output to use for uncertainty-based sample selection. When there are multiple outputs, active leanring loop selects one output as the selection target. Based on GP models predictive uncertainty for this output to select next training sample
- **Budget_Training_Sample** (int, default=50): Maximum number of training samples to use
- **NUM_INIT_SAMPLE** (int, default=10): Number of initial random samples
- **TOP_K** (int, default=1): Number of samples to select in each iteration
- **gp_config_path** (str, optional): Path to YAML configuration file

**Returns:**

- **GPModel**: Trained Gaussian Process model

### GPModel Class

Wrapper class for multi-output Gaussian Process regression.

```python
from ALGP.gp_model_module import GPModel

model = GPModel(
    x_training,      # Training features
    y_training,      # Training targets
    gp_config_path   # Config path (optional)
)

# Make predictions
y_mean, y_std = model.predict(x_test, return_std=True)
```

**Methods:**

- **fit(x_train, y_train)**: Fit the GP model to training data
- **predict(x_test, return_std=False)**: Make predictions
  - If `return_std=True`: Returns `(y_mean, y_std)`
  - If `return_std=False`: Returns `y_mean` only

## Active Learning Strategy

ALGP uses an uncertainty-based sampling strategy:

1. **Initialization**: Randomly select `NUM_INIT_SAMPLE` samples from the training pool
2. **Iteration**: Repeat until budget is reached:
   - Train GP model on selected samples
   - Predict on remaining samples
   - Select `TOP_K` samples with highest prediction uncertainty (std)
   - Add selected samples to training set
3. **Final Model**: Train on all selected samples

The `ActiveLearning_Target_Index` parameter determines which output's uncertainty is used for sample selection in multi-output scenarios.

## Project Structure

```
ALGP_package/
├── ALGP/                          # Main package
│   ├── __init__.py              # Package initialization
│   ├── ALGP_module.py           # Main ALGP function
│   ├── ALGP_loop.py             # Active learning loop
│   ├── gp_model_module.py       # GPModel wrapper class
│   ├── get_one_gp_model_module.py # GP model factory
│   ├── eval_module.py           # Evaluation utilities
│   ├── scikit_regression_metrics.py # Metrics computation
│   └── utils.py                 # Utility functions
├── configs/                      # Configuration files
│   └── gp-l_bfgs_v01.yaml       # Default GP config
├── scripts/                      # Example scripts
│   ├── toy_example.py           # Toy example
│   ├── train_ALGP_example.py    # Training example
│   ├── load_data.py             # Data loading
│   └── data.csv                 # Example dataset
├── pyproject.toml               # Package configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Requirements

- Python >= 3.7
- scikit-learn
- omegaconf
  
Install via:
```bash
pip install -r requirements.txt
```
<!-- 
## License

[Add your license information here]

## Citation

If you use this package in your research, please cite:

```
[Add citation information here]
```

## Contact

[Add contact information here]
 -->
