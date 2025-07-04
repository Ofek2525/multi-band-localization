import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

Z_EXP = "/home/ofekshis/multi-band-localization/z_exp"

# All no NN DATA
no_nn_path = f"{Z_EXP}/2025-06-29_20:15#more_layers#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"

# All NN DATA per SNR:
nn_m15snr_path = f"{Z_EXP}/2025-06-29_20:26#for_paper1#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
nn_m10snr_path = f"{Z_EXP}/2025-06-29_20:26#for_paper1#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
nn_m5snr_path = f"{Z_EXP}/2025-06-29_20:26#for_paper1#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
nn_0snr_path = f"{Z_EXP}/2025-06-29_20:26#for_paper1#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
nn_5snr_path = f"{Z_EXP}/2025-06-29_21:34#for_paper1#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"
nn_10snr_path = f"{Z_EXP}/2025-06-30_01:05#for_paper1#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=10.0dBm"

nn_data_paths = [nn_m15snr_path, nn_m10snr_path, nn_m5snr_path, nn_0snr_path, nn_5snr_path, nn_10snr_path]
snr_list = [-15, -10, -5, 0, 5, 10]
bands = ['6GHz (no NN)', '12GHz (no NN)', '18GHz (no NN)', '24GHz (no NN)']

def csv_to_list(ue_num, samples_only_per_snr_FLAG):
    results = []
    avg_error = []

    # NN model
    if samples_only_per_snr_FLAG:
        for path, snr in zip(nn_data_paths, snr_list):
            csv_filename = f"{path}/error_metrics_vs_input_power_{ue_num}UEs.csv"
            df = pd.read_csv(csv_filename)
            val = df.loc[(df['Input Power [dBm]'] == snr) & (df['Band'] == 'Multiband (with NN)'), 'Avg Error [m]'].iloc[0]
            avg_error.append(val)
    else: # Taking the best results regardless to the training SNR
        num_of_SNRs = len(snr_list)
        avg_error = [10000] * num_of_SNRs  # start with a very big num
        for i, path in enumerate(nn_data_paths):
            csv_filename = f"{path}/error_metrics_vs_input_power_{ue_num}UEs.csv"
            df = pd.read_csv(csv_filename)
            for snr_idx, snr in enumerate(snr_list):
                val = df.loc[(df['Input Power [dBm]'] == snr) & (df['Band'] == 'Multiband (with NN)'), 'Avg Error [m]'].iloc[0]
                if val < avg_error[snr_idx]:
                    avg_error[snr_idx] = val
    results.append(avg_error)

    # NO NN MUSIC
    csv_filename = f"{no_nn_path}/error_metrics_vs_input_power_{ue_num}UEs.csv"
    df = pd.read_csv(csv_filename)
    for band in bands:
        avg_error = []
        for snr in snr_list:
            val = df.loc[(df['Input Power [dBm]'] == snr) & (df['Band'] == band), 'Avg Error [m]'].iloc[0]
            avg_error.append(val)
        results.append(avg_error)
    
    # Avg MultiBeamformer(no NN)
    if ue_num == 1:
        avg_error = []
        for snr in snr_list:
            val = df.loc[(df['Input Power [dBm]'] == snr) & (df['Band'] == 'Avg MultiBeamformer(no NN)'), 'Avg Error [m]'].iloc[0]
            avg_error.append(val)
        results.append(avg_error)
    
    return results

    

def plot_MultiBandNet_and_music_singal_band(ue_num, samples_only_per_snr_FLAG=True):
    avg_errors = csv_to_list(ue_num, samples_only_per_snr_FLAG)

    plt.figure(figsize=(10, 6))

    # MultiBandNet (NN) - black, thick, solid
    plt.plot(
        snr_list,
        avg_errors[0],
        label='MultiBandNet (NN)',
        color='black',
        linestyle='solid',
        linewidth=2.5,
        marker='o',
        markersize=6
    )

    # Define styles and colors for MUSIC bands
    markers = ['s', '^', 'x', 'D']
    linestyles = ['dotted', 'dashed', 'dashdot', (0, (1, 2))]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, band in enumerate(bands):
        plt.plot(
            snr_list,
            avg_errors[i+1],
            label=f"MUSIC {band}",
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=2,
            marker=markers[i],
            markersize=5
        )

    # Optional: Avg MultiBeamformer (no NN)
    if ue_num == 1:
        plt.plot(
            snr_list,
            avg_errors[-1],
            label="Avg MultiBeamformer (no NN)",
            color='tab:purple',
            linestyle='solid',
            linewidth=2,
            marker='v',
            markersize=5
        )

    plt.title(f"Avg Localization Error vs SNR (UEs={ue_num})")
    plt.xlabel("Transmission power [dBm]")
    plt.ylabel("Avg euclidean distance Error [m]")
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    output_path = f"localization_error_vs_snr_{ue_num}UEs.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    

if __name__ == "__main__":
    for i in [1, 2]:
        plot_MultiBandNet_and_music_singal_band(i, samples_only_per_snr_FLAG=False)