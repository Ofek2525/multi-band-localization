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

## Usage

### Configuration

Before running, configure the system parameters in `python_code/exp_params.py`:

- **Frequency bands**: `fc = [6000, 12000, 18000, 24000]` (MHz)
- **Number of antennas**: `Nr = [4, 8, 16, 32]` (per band)
- **Number of subcarriers**: `K = [20, 20, 20, 20]` (per band)
- **Bandwidth**: `BW = [4, 4, 4, 4]` (MHz)
- **Input power**: `input_power = 10` (dBm)
- **Number of samples**: `NS = 50`
- **Algorithm**: `alg = 'MUSIC'`

### Training

Train a multi-band model:

```bash
cd python_code
python train.py
```

Train with custom parameters (for job arrays):

```bash
python train.py <input_power> <learning_rate> <batch_size> <tau> <NS>
```

Example:
```bash
python train.py 5.0 0.0008 16 4 50
```

Training parameters can also be modified in `train.py`:
- `learning_rate`: Learning rate (default: 1e-03)
- `batch_size`: Batch size (default: 16)
- `data_samples`: Number of training samples (default: 150000)
- `ues_num`: Number of UEs per sample (default: 2)
- `band`: 0 for multi-band, 1-4 for single-band (6G, 12G, 18G, 24G)

### Testing and Evaluation

#### Single Sample Test

Test localization on a specific UE position:

```bash
python main.py
```

Modify `ues_pos` in `main.py` to test different positions. The script will:
- Load a trained model
- Generate channel measurements
- Estimate AoA and ToA
- Compute localization error
- Generate visualization plots

#### Comprehensive Evaluation

Run comprehensive tests comparing multi-band vs single-band approaches:

```bash
python test.py
```

Or use the evaluation functions programmatically:

```python
from test import compare_MultiBandNet_to_music_singal_band

model_path = "z_exp/your_experiment/model_params.pth"
input_power_values = [-15, -10, -5, 0, 5]  # dBm
compare_MultiBandNet_to_music_singal_band(
    num_users=2, 
    input_power_list=input_power_values, 
    model_path=model_path,
    BS_num="all"  # or specific BS number
)
```

### Running Experiments

The `big_tests.py` module provides additional evaluation functions:

```python
from big_tests import test_and_save, compare_MultiBandNet_to_MultiBeamformer

# Test and save results for different numbers of UEs
test_and_save(num_ues=1, input_power_values=[5.0], model_path=model_path, BS_num="all")
test_and_save(num_ues=2, input_power_values=[5.0], model_path=model_path, BS_num="all")
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

## Data

The project uses ray-tracing simulation data stored in CSV format:
- Training data: `resources/all_BSs/bs_*/train_*Ghz.csv`
- Test data: `resources/all_BSs/bs_*/test_*Ghz.csv`

Each CSV file contains channel parameters including:
- UE and BS positions
- Angle-of-Arrival (AoA)
- Time-of-Arrival (ToA)
- Channel gains

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Multi-Band Localization in the Upper Mid-Band Using Deep Learning},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2024},
  note={Add your paper details here}
}
```

**Note**: Please update the citation above with your actual publication details when available.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add contact information if desired]

## Acknowledgments

[Add acknowledgments if applicable]
