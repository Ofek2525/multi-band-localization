import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

ROOT = "/home/ofekshis/multi-band-localization"

# All no NN DATA
no_nn_path = fr"{ROOT}/z_exp/2025-08-03_21:46#for_paper_3ues#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"

# All NN DATA per SNR:
multi_m15snr_path = fr"{ROOT}/z_exp/2025-08-03_21:46#for_paper_3ues#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
multi_m10snr_path = fr"{ROOT}/z_exp/2025-08-03_21:46#for_paper_3ues#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
multi_m5snr_path = fr"{ROOT}/z_exp/2025-08-03_21:47#for_paper_3ues#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
multi_0snr_path = fr"{ROOT}/z_exp/2025-08-03_21:46#for_paper_3ues#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
multi_5snr_path = fr"{ROOT}/z_exp/2025-08-03_21:46#for_paper_3ues#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"


single_6G_m15snr_path = fr"{ROOT}/z_exp/2025-08-04_10:18#for_paper_3ues_6ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
single_6G_m10snr_path = fr"{ROOT}/z_exp/2025-08-04_10:18#for_paper_3ues_6ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
single_6G_m5snr_path = fr"{ROOT}/z_exp/2025-08-04_10:18#for_paper_3ues_6ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
single_6G_0snr_path = fr"{ROOT}/z_exp/2025-08-04_11:27#for_paper_3ues_6ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
single_6G_5snr_path = fr"{ROOT}/z_exp/2025-08-04_11:27#for_paper_3ues_6ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"


single_24G_m15snr_path = fr"{ROOT}/z_exp/2025-08-04_10:18#for_paper_3ues_24ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
single_24G_m10snr_path = fr"{ROOT}/z_exp/2025-08-04_10:18#for_paper_3ues_24ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
single_24G_m5snr_path = fr"{ROOT}/z_exp/2025-08-04_10:18#for_paper_3ues_24ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
single_24G_0snr_path = fr"{ROOT}/z_exp/2025-08-04_10:18#for_paper_3ues_24ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
single_24G_5snr_path = fr"{ROOT}/z_exp/2025-08-04_10:18#for_paper_3ues_24ghz#tau =4 lr=0.0007,batch=12,ues=3,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"


multi_data_paths = [multi_m15snr_path,multi_m10snr_path,multi_m5snr_path,multi_0snr_path,multi_5snr_path]
single_6G_data_paths = [single_6G_m15snr_path,single_6G_m10snr_path,single_6G_m5snr_path,single_6G_0snr_path,single_6G_5snr_path]
single_24G_data_paths = [single_24G_m15snr_path,single_24G_m10snr_path,single_24G_m5snr_path,single_24G_0snr_path,single_24G_5snr_path]
snr_list = [-15, -10, -5, 0, 5]


def csv_to_list(ue_num):
    results = []
    avg_error = []

    # multibandnet
    for path, snr in zip(multi_data_paths, snr_list):
            csv_filename = f"{path}/error_metrics_vs_input_power_{ue_num}UEs.csv"
            df = pd.read_csv(csv_filename)
            val = df.loc[(df['Input Power [dBm]'] == snr) & (df['Band'] == 'Multiband (with NN)'), 'Avg Error [m]'].iloc[0]
            if not val:
                 print("erorr: val not fond1")
            avg_error.append(val)    
    results.append(avg_error)
    avg_error = []

    for path, snr in zip(single_6G_data_paths, snr_list):
        csv_filename = f"{path}/error_metrics_vs_input_power_{ue_num}UEs.csv"
        df = pd.read_csv(csv_filename)
        val = df.loc[(df['Input Power [dBm]'] == snr) & (df['Band'] == 'Multiband (with NN)'), 'Avg Error [m]'].iloc[0]
        if not val:
                print("erorr: val not fond2")
        avg_error.append(val)    
    results.append(avg_error)
    avg_error = []

    for path, snr in zip(single_24G_data_paths, snr_list):
            csv_filename = f"{path}/error_metrics_vs_input_power_{ue_num}UEs.csv"
            df = pd.read_csv(csv_filename)
            val = df.loc[(df['Input Power [dBm]'] == snr) & (df['Band'] == 'Multiband (with NN)'), 'Avg Error [m]'].iloc[0]
            if not val:
                 print("erorr: val not fond3")
            avg_error.append(val)    
    results.append(avg_error)
    avg_error = []

    bands = ['6GHz (no NN)', '24GHz (no NN)'] if ue_num != 1 else ['6GHz (no NN)', '12GHz (no NN)', '18GHz (no NN)', '24GHz (no NN)']
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
        avg_error = []
    
    return results



def plot_MultiBandNet_and_singlebandnet(ue_num):
    avg_errors = csv_to_list(ue_num)

    plt.figure(figsize=(10, 6))

    # MultiBandNet (NN) - black, thick, solid
    plt.plot(
        snr_list,
        avg_errors[0],
        label='MultiBandNet',
        color='black',
        linestyle='solid',
        linewidth=2.5,
        marker='o',
        markersize=6
    )

    # Define styles and colors for MUSIC bands
    markers = ['s', 'D']
    linestyles = ['dotted', (0, (1, 2))]
    colors = ['tab:blue', 'tab:red']
    labels = ["SubspaceNet @ 6GHz","SubspaceNet @ 24GHz"]

    for i in range(len(labels)):
        plt.plot(
            snr_list,
            avg_errors[i+1],
            label=labels[i],
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=2,
            marker=markers[i],
            markersize=5
        )

    markers = ['^', 'P', '*', 'x'] if ue_num == 1 else ['^', 'x']
    linestyles = ['dashed', (0, (3, 1, 1, 1)), (0, (5, 1)), 'dashdot'] if ue_num == 1 else ['dashed', 'dashdot']
    colors = ['tab:orange', 'tab:brown', 'tab:cyan', 'tab:green'] if ue_num == 1 else ['tab:orange', 'tab:green']
    labels = ["MUSIC @ 6GHz", "MUSIC @ 12GHz", "MUSIC @ 18GHz", "MUSIC @ 24GHz"] if ue_num == 1 else ["MUSIC @ 6GHz", "MUSIC @ 24GHz"]

    for i in range(len(labels)):
        plt.plot(
            snr_list,
            avg_errors[i+3],
            label=labels[i],
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
            label="MultiBeamformer",
            color='tab:purple',
            linestyle='solid',
            linewidth=2,
            marker='v',
            markersize=5
        )

    #plt.title(f"Avg Localization Error vs SNR (UEs={ue_num})")
    plt.xlabel("Transmission power [dBm]")
    plt.ylabel("Avg euclidean distance Error [m]")
    plt.grid(True)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    output_path = f"localization_error_vs_snr_{ue_num}UEs(subspacenet).png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    

if __name__ == "__main__":
    for i in [3]:
        plot_MultiBandNet_and_singlebandnet(i)