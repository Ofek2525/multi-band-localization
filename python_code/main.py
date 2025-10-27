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
BS_num =12
model_path = r"z_exp/2025-09-17_19:12#back_to_k20#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"
if "model_params.pth" not in model_path:
    model_path = fr"{model_path}/model_params.pth"
model_path = fr"{ROOT_DIR}/{model_path}"
no_nn = 0
def main():

    # for cases when exp_parans are multiband:
    main_band =3
    # 1 for single band 6G with no net
    # 2 for single band 12G with no net
    # 3 for single band 18G with no net
    # 4 for single band 24G with no net

    bands = None
    problematic =[[[790, 0], [450, 145]], [[430, 20], [450, 145]], [[735, 25], [410, 195]], [[730, 30], [420, 100]], [[675, 50], [450, 145]], [[665, 55], [440, 80]], [[635, 70], [445, 155]], [[660, 75], [440, 80]], [[440, 80], [610, 85]], [[440, 80], [435, 180]], [[400, 85], [455, 175]], [[610, 85], [420, 100]], [[560, 110], [445, 155]], [[575, 115], [450, 165]], [[450, 145], [560, 155]], [[575, 145], [410, 195]]]
    ues_pos = np.array([[790,0]])#[[50,220],[580, 380]]#[245, 355]#[160, 215]#[100,90]#[240,370]#[120,125]#[50,15]  # transmitter UE position    
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