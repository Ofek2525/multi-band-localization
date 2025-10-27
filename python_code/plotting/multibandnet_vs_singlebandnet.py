import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# All no NN DATA
no_nn_path = fr"{ROOT}/z_exp/0important_copys/results_for_all_musics:NS=50,Tres=0.03"

# All NN DATA per SNR:
multi_m15snr_path = fr"z_exp/2025-10-15_16:45#retrain1#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
multi_m10snr_path = fr"z_exp/retest_K20/2025-06-29_20:26#for_paper1#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
multi_m5snr_path = fr"z_exp/retest_K20/2025-06-29_20:26#for_paper1#tau =4 lr=0.001,batch=20,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
multi_0snr_path = fr"z_exp/2025-10-15_16:45#retrain1#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
multi_5snr_path = fr"z_exp/2025-10-15_16:45#retrain1#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"


single_6G_m15snr_path = fr"z_exp/retest_K20/2025-07-24_16:30#for_paper6ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
single_6G_m10snr_path = fr"z_exp/retest_K20/2025-07-24_16:30#for_paper6ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
single_6G_m5snr_path = fr"z_exp/retest_K20/2025-07-24_17:27#for_paper6ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
single_6G_0snr_path = fr"z_exp/retest_K20/2025-07-24_17:28#for_paper6ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
single_6G_5snr_path = fr"z_exp/retest_K20/2025-07-24_18:56#for_paper6ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"


single_24G_m15snr_path = fr"z_exp/retest_K20/2025-07-24_16:28#for_paper24ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
single_24G_m10snr_path = fr"z_exp/retest_K20/2025-07-24_16:28#for_paper24ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
single_24G_m5snr_path = fr"z_exp/retest_K20/2025-07-24_16:28#for_paper24ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
single_24G_0snr_path = fr"z_exp/retest_K20/2025-07-24_16:28#for_paper24ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
single_24G_5snr_path = fr"z_exp/retest_K20/2025-07-24_16:28#for_paper24ghz#tau =4 lr=0.0007,batch=12,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"

# # All no NN DATA
# no_nn_path = fr"{ROOT}/z_exp/no_net_result"

# # All NN DATA per SNR:
# multi_m15snr_path = fr"z_exp/2025-09-17_19:14#back_to_k20#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
# multi_m10snr_path = fr"z_exp/2025-09-17_19:11#back_to_k20#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
# multi_m5snr_path = fr"z_exp/2025-09-17_19:11#back_to_k20#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
# multi_0snr_path = fr"z_exp/2025-09-18_20:56#back_to_k20#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
# multi_5snr_path = fr"z_exp/2025-09-17_19:12#back_to_k20#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"


# single_6G_m15snr_path = fr"z_exp/2025-09-18_21:03#back_to_k20_6ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
# single_6G_m10snr_path = fr"z_exp/2025-09-18_21:06#back_to_k20_6ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
# single_6G_m5snr_path = fr"z_exp/2025-09-18_21:08#back_to_k20_6ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
# single_6G_0snr_path = fr"z_exp/2025-09-18_21:10#back_to_k20_6ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
# single_6G_5snr_path = fr"z_exp/2025-09-25_13:08#back_to_k20_6ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"


# single_24G_m15snr_path = fr"z_exp/2025-09-18_22:15#back_to_k20_24ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-15.0dBm"
# single_24G_m10snr_path = fr"z_exp/2025-09-18_22:32#back_to_k20_24ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-10.0dBm"
# single_24G_m5snr_path = fr"z_exp/2025-09-18_22:35#back_to_k20_24ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=-5.0dBm"
# single_24G_0snr_path = fr"z_exp/2025-09-18_22:40#back_to_k20_24ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=0.0dBm"
# single_24G_5snr_path = fr"z_exp/2025-09-18_21:38#back_to_k20_24ghz#tau =1 lr=0.0008,batch=16,ues=2,k=[20, 20, 20, 20],Nr=[4, 8, 16, 32],fc=[6000, 12000, 18000, 24000],BW=[4, 4, 4, 4],NS=50,input_power=5.0dBm"


multi_data_paths = [multi_m15snr_path,multi_m10snr_path,multi_m5snr_path,multi_0snr_path,multi_5snr_path]
single_6G_data_paths = [single_6G_m15snr_path,single_6G_m10snr_path,single_6G_m5snr_path,single_6G_0snr_path,single_6G_5snr_path]
single_24G_data_paths = [single_24G_m15snr_path,single_24G_m10snr_path,single_24G_m5snr_path,single_24G_0snr_path,single_24G_5snr_path]
multi_data_paths = [fr"{ROOT}/" + p for p in multi_data_paths]
single_6G_data_paths = [fr"{ROOT}/" + p for p in single_6G_data_paths]
single_24G_data_paths = [fr"{ROOT}/" + p for p in single_24G_data_paths]
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

    bands = ['6GHz (no NN)', '24GHz (no NN)']
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

    markers =  ['^', 'x']
    linestyles =  ['dashed', 'dashdot']
    colors =  ['tab:orange', 'tab:green']
    labels =  ["MUSIC @ 6GHz", "MUSIC @ 24GHz"]

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
    plt.legend(fontsize=14)
    plt.tight_layout()

    output_path = f"localization_error_vs_snr_{ue_num}UEs(subspacenet).png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    

if __name__ == "__main__":
    for i in [1,2]:
        plot_MultiBandNet_and_singlebandnet(i)