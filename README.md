# Multi-Band Localization

Data-driven localization in the upper mid-band utilizing multiple frequency bands. This repository implements a deep learning-based approach for improving user equipment (UE) localization accuracy by leveraging signals from multiple frequency bands (6 GHz, 12 GHz, 18 GHz, and 24 GHz).

## Overview

This project presents a multi-band neural network architecture (`Multi_Band_SubSpaceNET`) that processes channel measurements from multiple frequency bands to enhance localization performance. The approach combines deep learning with traditional signal processing techniques (MUSIC algorithm) to estimate Angle-of-Arrival (AoA) and Time-of-Arrival (ToA), which are then used for UE position estimation.

### Key Features

- **Multi-band signal processing**: Utilizes 4 frequency bands (6, 12, 18, 24 GHz) simultaneously
- **Deep learning-enhanced covariance estimation**: Neural network improves covariance matrix estimation for MUSIC algorithm
- **Single-band and multi-band support**: Can operate with single or multiple frequency bands
- **Ray-tracing based channel simulation**: Uses realistic channel data from ray-tracing simulations
- **Comprehensive evaluation**: Compares multi-band approach against single-band MUSIC baseline

## Installation

### Prerequisites

- Python 3.10
- CUDA 12.1 (for GPU acceleration)
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-band-localization
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate pytorch_py310_cu121
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
multi-band-localization/
├── python_code/              # Main source code
│   ├── estimation/           # Neural networks and estimation algorithms
│   │   ├── multiband_net.py  # Multi-band neural network architecture
│   │   ├── net.py            # Single-band neural network
│   │   ├── music.py          # MUSIC algorithm implementation
│   │   ├── beamformer.py     # Beamforming algorithms
│   │   └── estimate.py      # Estimation utilities
│   ├── channel/              # Channel generation and loading
│   │   ├── channel_loader.py # Load channel data from CSV
│   │   └── generate_channel.py # Generate channel measurements
│   ├── utils/                # Utility functions
│   │   ├── bands_manipulation.py
│   │   └── learning_rate_schedule.py
│   ├── plotting/             # Visualization tools
│   ├── train.py              # Training script
│   ├── test.py               # Testing and evaluation
│   ├── main.py               # Main entry point for single tests
│   └── exp_params.py         # Experiment parameters
├── resources/                # Data files
│   ├── all_BSs/             # Base station channel data
│   └── raytracing/          # Ray-tracing simulation data
├── results/                  # Output plots and results
├── z_exp/                    # Experiment outputs and saved models
└── environment.yml           # Conda environment specification
```

## Key Components

### Multi-Band SubSpaceNET

The `Multi_Band_SubSpaceNET` architecture consists of:
- **Per-band encoders**: Separate encoders for each frequency band (6k, 12k, 18k, 24k)
- **Feature fusion**: Concatenation of encoded features from all bands
- **Decoder**: Reconstructs improved covariance matrix for MUSIC algorithm

### Estimation Pipeline

1. **Channel generation**: Generate or load channel measurements from ray-tracing data
2. **Covariance estimation**: Compute autocorrelation matrices (with or without neural network enhancement)
3. **AoA/ToA estimation**: Apply MUSIC algorithm to estimate angles and delays
4. **Position estimation**: Convert AoA/ToA to 2D positions using geometric relationships