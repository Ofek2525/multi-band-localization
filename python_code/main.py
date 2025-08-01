import numpy as np
import torch
from estimation.net import single_nurone, SubSpaceNET
from estimation.multiband_net import Multi_Band_SubSpaceNET, Encoder_6k, Encoder_12k, Encoder_18k, Encoder_24k, Decoder
from utils.bands_manipulation import get_bands_from_conf, Band
from exp_params import seed, tau,K, Nr, fc, BW, alg, aoa_res, T_res, plot_estimation_results
from plotting.map_plot import plot_angle_time
from test import test_1sample
from dir_definitions import ROOT_DIR

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BS_num =13
model_path = r"z_exp/2025-07-06_19:24#main_band_24G#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10dBm"
if "model_params.pth" not in model_path:
    model_path = fr"{model_path}/model_params.pth"
model_path = fr"{ROOT_DIR}/{model_path}"
no_nn = 0
def main():

    # for cases when exp_parans are multiband:
    main_band = 1
    # 1 for single band 6G with no net
    # 2 for single band 12G with no net
    # 3 for single band 18G with no net
    # 4 for single band 24G with no net

    bands = None
    ues_pos = np.array([[465,0]])#[245, 355]#[160, 215]#[100,90]#[240,370]#[120,125]#[50,15]  # transmitter UE position    
    if no_nn == 0:
        if len(fc) == 1:
            model = SubSpaceNET().to(DEVICE)
        elif len(fc) == 4:
            model = Multi_Band_SubSpaceNET(tau).to(DEVICE)
        else:
            print("error with params")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    else:
        model = single_nurone().to(DEVICE)
        if len(fc) != 1:
            bands = [get_bands_from_conf(fc, Nr, K, BW)[main_band - 1]]
    test_1sample(model, ues_pos, toPlot=True,name=r"results/AOA_and_delay_est_net.png",zoom =False, bands=bands,BS_num =BS_num)


if __name__ == "__main__":
    main()