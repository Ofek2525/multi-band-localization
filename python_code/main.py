import numpy as np
import torch
import os
import sys
from estimation.net import single_nurone, SubSpaceNET
from estimation.multiband_net import Multi_Band_SubSpaceNET, Encoder_6k, Encoder_12k, Encoder_18k, Encoder_24k, Decoder
from utils.bands_manipulation import get_bands_from_conf, Band
from exp_params import seed, tau, K, Nr, fc, BW, alg, aoa_res, T_res, plot_estimation_results
from plotting.map_plot import plot_angle_time
from test import test_1sample
from dir_definitions import ROOT_DIR

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configuration: Can be set via environment variable or command line argument
BS_num = int(os.getenv("BS_NUM", "12"))
no_nn = int(os.getenv("NO_NN", "0"))

# Model path: Use command line argument if provided, otherwise environment variable, otherwise None
if len(sys.argv) > 1:
    model_path = sys.argv[1]
elif os.getenv("MODEL_PATH"):
    model_path = os.getenv("MODEL_PATH")
else:
    # Default example path - users should set MODEL_PATH or pass as argument
    model_path = None
    print("Warning: No model path provided. Set MODEL_PATH environment variable or pass as command line argument.")
    print("Usage: python python_code/main.py <model_path> [BS_num] [no_nn]")
    sys.exit(1)

if model_path and "model_params.pth" not in model_path:
    model_path = fr"{model_path}/model_params.pth"
if model_path and not os.path.isabs(model_path):
    model_path = fr"{ROOT_DIR}/{model_path}"

# Optional: Override BS_num and no_nn from command line
if len(sys.argv) > 2:
    BS_num = int(sys.argv[2])
if len(sys.argv) > 3:
    no_nn = int(sys.argv[3])
def main():
    """
    Main function to run localization estimation.
    
    Model path should be provided via:
    - Command line argument: python main.py <model_path> [BS_num] [no_nn]
    - Environment variable: MODEL_PATH
    """
    # for cases when exp_params are multiband:
    main_band = 3
    # 1 for single band 6G with no net
    # 2 for single band 12G with no net
    # 3 for single band 18G with no net
    # 4 for single band 24G with no net

    bands = None
    # Example UE positions for testing
    ues_pos = np.array([[790, 0]])  # transmitter UE position
    
    if no_nn == 0:
        if len(fc) == 1:
            model = SubSpaceNET().to(DEVICE)
        elif len(fc) == 4:
            model = Multi_Band_SubSpaceNET(tau).to(DEVICE)
        else:
            raise ValueError(f"Unsupported number of frequency bands: {len(fc)}. Expected 1 or 4.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    else:
        model = single_nurone().to(DEVICE)
        if len(fc) != 1:
            bands = [get_bands_from_conf(fc, Nr, K, BW)[main_band - 1]]
    
    test_1sample(model, ues_pos, toPlot=True, name=os.path.join(ROOT_DIR, "results", "AOA_and_delay_est_net.png"), 
                 zoom=False, bands=bands, BS_num=BS_num)


if __name__ == "__main__":
    main()