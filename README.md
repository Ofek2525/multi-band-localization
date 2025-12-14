# Multi-Band Localization

Data-driven localization in the upper mid-band utilizing multiple frequency bands.

## Overview

This research project implements neural network-based improvements to traditional MUSIC algorithm for angle-of-arrival (AOA) and time-of-arrival (TOA) estimation in wireless communication scenarios. The system uses PyTorch to train SubSpaceNET and Multi_Band_SubSpaceNET models that operate across multiple frequency bands (6GHz, 12GHz, 18GHz, 24GHz).

## Installation

### Requirements
- Python 3.10
- CUDA 12.1 (for GPU support)
- Conda

### Setup Environment
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate pytorch_py310_cu121
```

## Quick Start

### Basic Localization Test
```bash
# Run single localization example
python python_code/main.py
```

### Training
```bash
# Single band training
python python_code/train.py

# Training with custom parameters [input_power, learning_rate, batch_size, tau, NS]
# NS is optional; if omitted, uses default from exp_params.py
python python_code/train.py -10 0.001 20 4 50
```
```

### Comprehensive Testing
```bash
# Run full test suite with comparisons
python python_code/test.py
```

## Configuration

Key parameters can be modified in `python_code/exp_params.py`:

- **K**: Number of subcarriers per band `[20,20,20,20]`
- **Nr**: Number of ULA elements `[4,8,16,32]` 
- **fc**: Carrier frequencies in MHz `[6000,12000,18000,24000]`
- **BW**: Bandwidth per band in MHz `[4,4,4,4]`
- **tau**: Time correlation parameter (default: 4)
- **alg**: Algorithm choice ('MUSIC', 'Beamformer', 'MultiBeamformer')

## Project Structure

```
├── python_code/           # Main source code
│   ├── estimation/        # Neural network models and estimation algorithms
│   ├── channel/          # Channel generation and data loading
│   ├── utils/            # Utility functions and data manipulation
│   └── plotting/         # Visualization tools
├── resources/            # Dataset and simulation data
│   ├── raytracing/       # Ray tracing simulation results
│   └── all_BSs/         # Multi-base station datasets
├── z_exp/               # Experiment outputs and trained models
└── results/             # Generated plots and analysis
```

## Usage Examples

### Single vs Multi-band Comparison
The system supports both single-band and multi-band operation:

- **Single band**: Uses one frequency band with SubSpaceNET
- **Multi-band**: Uses four frequency bands with Multi_Band_SubSpaceNET

### Model Loading
Trained models are automatically saved in timestamped directories under `z_exp/` with descriptive names containing hyperparameters.

## Results

The system generates:
- AOA and TOA estimation plots
- Learning curves during training
- Performance comparison charts
- Error distribution histograms
