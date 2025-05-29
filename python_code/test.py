import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import linear_sum_assignment
import os

from channel.generate_channel import get_channel, random_pos_ues_channel
from channel.channel_loader import get_ue_info_by_row
from estimation.estimate import estimate_evaluation
from estimation.net import single_nurone, SubSpaceNET
from estimation.multiband_net import Multi_Band_SubSpaceNET, Encoder_6k, Encoder_12k, Encoder_18k, Encoder_24k, Decoder
from utils.bands_manipulation import get_bands_from_conf, Band
from exp_params import seed, K, Nr, fc, BW, alg, aoa_res, T_res, plot_estimation_results
from plotting.map_plot import plot_angle_time
from dir_definitions import RAYTRACING_DIR
from utils.check_if_close import too_close

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = r"z_exp/2025-05-24_23:56#deffrent_postprocess#ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=10dBm/model_params.pth"

def test_multi_ue(band: Band, num_users: int, model_path):
    assert num_users <= 2
    torch.manual_seed(seed)
    no_nn = 0
    
    if no_nn == 0:
        if len(fc) == 1:
            model = SubSpaceNET().to(DEVICE)
        elif len(fc) == 4:
            model = Multi_Band_SubSpaceNET().to(DEVICE)
        else:
            print("error with params")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    else:
        model = single_nurone().to(DEVICE) 

    band_freq_file_in_k = int(band.fc / 1000)
    csv_file = rf"{RAYTRACING_DIR}/{band_freq_file_in_k}000/LOS_bs1_{band_freq_file_in_k}k_test.csv"
    df = pd.read_csv(csv_file)
    num_rows = len(df)

    sum_error = 0
    count_combinations = 0
    error_list = []
    problematic_locs = []
    # Create all combinations of UE indices
    ue_indices = list(range(num_rows))
    comb_indices = list(combinations(ue_indices, num_users))
    
    for comb in comb_indices:
        ue_locs = []
        for row_num in comb:
            ue_info = get_ue_info_by_row(csv_file, row_num)
            ue_locs.append(ue_info['ue_loc'])
        ue_locs_array = np.array(ue_locs)
        if too_close(ue_locs_array,50): continue
        temp_error = test_1sample(model, ue_locs_array)
        sum_error += temp_error
        error_list.append(temp_error)
        count_combinations += 1
        if temp_error > 70: problematic_locs.append(ue_locs)

    print("-"*40)
    print(f"Total {count_combinations} {num_users}-UE combinations tested.")
    print(f"AVG error dist: {sum_error / count_combinations:.3f} [m]")
    print(f"median error:{np.median(error_list):.3f}[m]")
    print("-"*40)
    print(problematic_locs)

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(error_list, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Error Distribution - {num_users} UEs")
    plt.xlabel("Error distance [m]")
    plt.ylabel("Number of combinations")
    plt.grid(True)
    plt.tight_layout()
    if len(fc) == 1:
        plt.savefig(f"error_hist_{band_freq_file_in_k}k_{num_users}UEs.png")
    else:
        plt.savefig(f"error_hist_multiband_{num_users}UEs.png")
    plt.close()



def sweep_input_power_single_band_multi_ue(band: Band, num_users: int, input_power_list: list[float],model_path):
    assert num_users <= 2
    torch.manual_seed(seed)
    band_freq_file_in_k = int(band.fc / 1000)
    csv_file = rf"{RAYTRACING_DIR}/{band_freq_file_in_k}000/LOS_bs1_{band_freq_file_in_k}k_test.csv"
    df = pd.read_csv(csv_file)
    num_rows = len(df)

    ue_indices = list(range(num_rows))
    comb_indices = list(combinations(ue_indices, num_users))

    results = {}

    for no_nn in [1, 0]:
        if no_nn == 0:
            model = SubSpaceNET().to(DEVICE)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            print("-"*40)
            print(" "*15+"using our net"+" "*15)
        else:
            model = single_nurone().to(DEVICE)
            print("-"*40)
            print(" "*15+"without our net"+" "*15)

        avg_errors = []
        median_errors = []

        for input_power in input_power_list:
            total_error = 0
            valid_comb_count = 0
            error_list = []

            for comb in comb_indices:
                ue_locs = []
                for row in comb:
                    ue_info = get_ue_info_by_row(csv_file, row)
                    ue_locs.append(ue_info['ue_loc'])
                ue_locs_array = np.array(ue_locs)
                if too_close(ue_locs_array,50): continue
                error = test_1sample(model, ue_locs_array, input_power=input_power, toPrint=False, bands=bands)
                total_error += error
                error_list.append(error)
                valid_comb_count += 1

            avg_error = total_error / valid_comb_count
            avg_errors.append(avg_error)
            median_errors.append(np.median(error_list))
            print("-"*40)
            print(f"input power:{input_power}[dBm]")
            print(f"Total {valid_comb_count} {num_users}-UE combinations tested.")
            print(f"AVG error dist: {avg_error:.3f} [m]")
            print(f"median error:{median_errors[-1]:.3f}[m]")
            

        results[no_nn] = (avg_errors, median_errors)
        

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(input_power_list, results[0][0], marker='o', color='blue', label='Avg (with NN)')
    plt.plot(input_power_list, results[1][0], marker='o', color='orange', label='Avg (no NN)')
    
    plt.title(f"Error vs Input Power - {num_users} UEs")
    plt.xlabel("Input Power [dBm]")
    plt.ylabel("Error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save to same folder as model_path
    save_dir = os.path.dirname(model_path)
    plt.savefig(os.path.join(save_dir, f"Avg_error_vs_input_power_{num_users}UEs.png"))
    plt.savefig(os.path.join("results", f"Avg_error_vs_input_power_{num_users}UEs.png"))
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.plot(input_power_list, results[0][1], marker='s', color='navy', label='Median (with NN)')
    plt.plot(input_power_list, results[1][1], marker='s', color='red', label='Median (no NN)')

    plt.title(f"Error vs Input Power - {num_users} UEs")
    plt.xlabel("Input Power [dBm]")
    plt.ylabel("Error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save to same folder as model_path
    save_dir = os.path.dirname(model_path)
    plt.savefig(os.path.join(save_dir, f"Median_error_vs_input_power_{num_users}UEs.png"))
    plt.savefig(os.path.join("results", f"Median_error_vs_input_power_{num_users}UEs.png"))
    plt.close()



def sweep_input_power_multi_band_multi_ue(num_users: int, input_power_list: list[float], model_path):
    assert num_users <= 3
    torch.manual_seed(seed)
    band_freq_file_in_k = 6
    csv_file = rf"{RAYTRACING_DIR}/{band_freq_file_in_k}000/LOS_bs1_{band_freq_file_in_k}k_test.csv"
    df = pd.read_csv(csv_file)
    num_rows = len(df)

    ue_indices = list(range(num_rows))
    comb_indices = list(combinations(ue_indices, num_users))

    results = {}

    for band in [0, 1, 2, 3, 4]:
        # 0 for multiband with net
        # 1 for single band 6G with no net
        # 2 for single band 12G with no net
        # 3 for single band 18G with no net
        # 4 for single band 24G with no net
        if band == 0: 
            model = Multi_Band_SubSpaceNET().to(DEVICE)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.eval()
            bands = None
            print("-"*40)
            print(" "*15+"using our net"+" "*15)
        else:
            model = single_nurone().to(DEVICE)
            bands = [get_bands_from_conf(fc, Nr, K, BW)[band - 1]]
            print("-"*40)
            print(" "*15+f"without our net {band}"+" "*15)

        avg_errors = []
        median_errors = []

        for input_power in input_power_list:
            total_error = 0
            valid_comb_count = 0
            error_list = []
            temp_ue_row = 0
            problematic_locs = []

            for comb in comb_indices:
                if num_users > 2:
                    if temp_ue_row % 10 != 0:
                        temp_ue_row += 1
                        continue
                    temp_ue_row += 1
                ue_locs = []
                for row in comb:
                    ue_info = get_ue_info_by_row(csv_file, row)
                    ue_locs.append(ue_info['ue_loc'])
                ue_locs_array = np.array(ue_locs)
                if too_close(ue_locs_array,50): continue
                error = test_1sample(model, ue_locs_array, input_power=input_power, toPrint=False, bands=bands)
                total_error += error
                error_list.append(error)
                valid_comb_count += 1
                if input_power == 10 and error > 70: problematic_locs.append(ue_locs)

            avg_error = total_error / valid_comb_count
            avg_errors.append(avg_error)
            median_errors.append(np.median(error_list))
            print("-"*40)
            print(f"input power:{input_power}[dBm]")
            print(f"Total {valid_comb_count} {num_users}-UE combinations tested.")
            print(f"AVG error dist: {avg_error:.3f} [m]")
            print(f"median error:{median_errors[-1]:.3f}[m]")
            print(problematic_locs)
            

        results[band] = (avg_errors, median_errors)

      
        

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(input_power_list, results[4][0], marker='o', color='yellow', label='Avg 24GHz (no NN)')
    plt.plot(input_power_list, results[3][0], marker='o', color='blue', label='Avg 18GHz (no NN)')
    plt.plot(input_power_list, results[2][0], marker='o', color='green', label='Avg 12GHz (no NN)')
    plt.plot(input_power_list, results[1][0], marker='o', color='orange', label='Avg 6GHz (no NN)')
    plt.plot(input_power_list, results[0][0], marker='o', color='red', label='Avg multiband (with NN)')
    
    plt.title(f"Error vs Input Power - {num_users} UEs")
    plt.xlabel("Input Power [dBm]")
    plt.ylabel("Error [m]")
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    # Save to same folder as model_path
    save_dir = os.path.dirname(model_path)
    plt.savefig(os.path.join(save_dir, f"Avg_error_vs_input_power_{num_users}UEs.png"))
    plt.savefig(os.path.join("results", f"Avg_error_vs_input_power_{num_users}UEs.png"))
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.plot(input_power_list, results[4][1], marker='s', color='yellow', label='Median 24GHz (no NN)')
    plt.plot(input_power_list, results[3][1], marker='s', color='blue', label='Median 18GHz (no NN)')
    plt.plot(input_power_list, results[2][1], marker='s', color='green', label='Median 12GHz (no NN)')
    plt.plot(input_power_list, results[1][1], marker='s', color='orange', label='Median 6GHz (no NN)')
    plt.plot(input_power_list, results[0][1], marker='s', color='red', label='Median multiband (with NN)')
    
    

    plt.title(f"Error vs Input Power - {num_users} UEs")
    plt.xlabel("Input Power [dBm]")
    plt.ylabel("Error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')
    # Save to same folder as model_path
    save_dir = os.path.dirname(model_path)
    plt.savefig(os.path.join(save_dir, f"Median_error_vs_input_power_{num_users}UEs.png"))
    plt.savefig(os.path.join("results", f"Median_error_vs_input_power_{num_users}UEs.png"))
    plt.close()


def test_1sample(model, ues_pos, toPlot=False, toPrint=True, name=None,zoom = False, input_power=None, bands=None):
    if not name:
        name = fr"results/test {list(ues_pos)}.png"
    """---------------------------------- CONFIG ---------------------------------------------"""
    # System parameters
    num_of_ues = len(ues_pos)
    bs_pos = np.array([280.863,473.291])
    '''----------------------------------------------------------------------------------------'''
    # random seed for run
    torch.manual_seed(seed)
    assert ues_pos.shape[1] == 2
    if bands is None:
        bands = get_bands_from_conf(fc, Nr, K, BW)
    if toPrint == True:
        print(f"UE: {ues_pos}") 
    # ------------------------------------- #
    # Physical Parameters' Estimation Phase #
    # ------------------------------------- #
    per_band_y, per_band_data = [], []
    # for each frequency sub-band
    for j, band in enumerate(bands):
        # generate the channel
        y, data = get_channel(ues_pos, band, input_power)
        per_band_y.append(y)
        per_band_data.append(data)
    multiband = "MULTI" if len(per_band_y) > 1 else "SINGLE"
    with torch.no_grad():
        estimations, spectrum,aoa_grid,times_grid = estimate_evaluation(alg, multiband, per_band_y, bands, model, num_of_ues)

    estimated_positions = []
    true_aoas = []
    true_toas = []
    for ue_indx, ue_data in enumerate(per_band_data[0]):
        if toPrint == True:
            print(f"ue{ue_indx + 1}: [AOA,TOA] =[{ue_data['aoa'][0]},{ue_data['toa'][0]}] ")
        true_aoas.append(ue_data['aoa'][0])
        true_toas.append(ue_data['toa'][0])
        AOA, TOA = estimations[0, ue_indx, :]
        c = 299.792
        h = 95
        r = np.sqrt((TOA * c) ** 2 - h ** 2) if TOA * c > h else 0
        est_pos = bs_pos + r * np.array([np.sin(np.deg2rad(AOA)), -np.cos(np.deg2rad(AOA))])
        estimated_positions.append(est_pos)

    estimated_positions = np.array(estimated_positions)  # shape (N, 2)

    # Compute pairwise distances between each estimated and true position
    num_ues = len(ues_pos)
    distance_matrix = np.zeros((num_ues, num_ues))
    for i in range(num_ues):
        for j in range(num_ues):
            distance_matrix[i, j] = np.linalg.norm(ues_pos[i] - estimated_positions[j])

    # Find the best assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    total_error = distance_matrix[row_ind, col_ind].sum()

    if toPrint == True:
        print(f"TOA and AOA estimations: {estimations}")
        print(f"Estimated positions (matched):\n{estimated_positions[col_ind]}")
        print(f"Average error per user: {total_error / num_ues}")
        print("-"*40)

    if toPlot:
        plot_angle_time(np.array(spectrum[0, :, :].to("cpu")), aoa_grid, times_grid, ues_pos,true_aoas,true_toas,name,zoom)
    # return error_distance
    return total_error / num_ues


if __name__ == "__main__":
    print("running...")
    num_ue = 1
    bands = get_bands_from_conf(fc, Nr, K, BW)
    no_sweep = 0 # sweep snr - 0 / one snr - 1
    if no_sweep == 1:
        test_multi_ue(bands[0], num_ue,model_path)
    elif no_sweep == 0:
        input_power_values = [-5, 0, 5, 10]
        sweep_input_power_multi_band_multi_ue(num_ue, input_power_values,model_path)
    else:
        print("only 0/1")
    