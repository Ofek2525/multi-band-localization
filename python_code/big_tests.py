import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from exp_params import seed, K, Nr, fc, BW, alg, aoa_res, T_res, plot_estimation_results
from test import sweep_input_power
from plotting.tests_plot import plots_of_compare_MultiBandNet_to_MultiBeamformer,plots_of_test_and_save
from dir_definitions import ROOT_DIR

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = r"z_exp/2025-06-29_20:15#more_layers#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
if "model_params.pth" not in model_path:
    model_path = fr"{model_path}/model_params.pth"
model_path = fr"{ROOT_DIR}/{model_path}"

def compare_MultiBandNet_to_MultiBeamformer(input_power_list: list[float], model_path,BS_num=1):
    #compare the mse vs transmition power of our multi subband method to music with singal subband.
    num_users = 1
    results = {}

    for method in [0, 1, 2, 3, 4, 5]:
        # 0 for multiband with net
        # 1 for single band 6G with no net
        # 2 for single band 12G with no net
        # 3 for single band 18G with no net
        # 4 for single band 24G with no net
        # 5 for MultiBeamformer
        if method == 0: 
            no_NN = 0
            alg = 'MUSIC'
            print("-"*40)
            print(" "*15+"using our net"+" "*15)
            band = 0
        elif method == 5:
            no_NN = 1 
            alg = 'MultiBeamformer'
            print("-"*40)
            print(" "*15+f"using multi subband beamformer"+" "*15)
            band = 0 
        else:
            no_NN = 1 
            alg = 'MUSIC'
            band = method
            print("-"*40)
            print(" "*15+f"without our net {fc[band-1]//1000}G"+" "*15)
        avg_errors, median_errors = sweep_input_power(model_path,num_users,input_power_list,band,no_NN,alg,BS_num=BS_num)
        results[method] = (avg_errors, median_errors) 
    # Plot
    plots_of_compare_MultiBandNet_to_MultiBeamformer(results,input_power_list,num_users,model_path)


def test_and_save(num_users,input_power_list: list[float], model_path,BS_num="all"):
    #mse vs transmition power of our multi subband method.
    results = {}
    no_NN = 0
    alg = 'MUSIC'
    print("-"*40)
    print(" "*15+"using our net"+" "*15)
    band = 0
    avg_errors, median_errors = sweep_input_power(model_path,num_users,input_power_list,band,no_NN,alg,BS_num=BS_num)
    results[0] = (avg_errors, median_errors) 
    # Plot
    plots_of_test_and_save(results,input_power_list,num_users,model_path)

if __name__ == "__main__":
    BS_num = "all"
    input_power_values = [-15,-10,-5, 0, 5, 10]
    compare_MultiBandNet_to_MultiBeamformer(input_power_values, model_path,BS_num=BS_num)
    #test_and_save(1,input_power_values,model_path,BS_num)