import matplotlib.pyplot as plt
import numpy as np
import os

def plots_of_MultiBandNet_to_music_singal_band(results,input_power_list,num_users,model_path):
    plt.figure(figsize=(8, 5))
    plt.plot(input_power_list, results[4][0], marker='o', color='yellow', label='Avg 24GHz (no NN)')
    plt.plot(input_power_list, results[3][0], marker='o', color='blue', label='Avg 18GHz (no NN)')
    plt.plot(input_power_list, results[2][0], marker='o', color='green', label='Avg 12GHz (no NN)')
    plt.plot(input_power_list, results[1][0], marker='o', color='orange', label='Avg 6GHz (no NN)')
    plt.plot(input_power_list, results[0][0], marker='o', color='red', label='Avg multiband (with NN)')
    
    plt.title(f"Error vs Transmitted Power - {num_users} UEs")
    plt.xlabel("Transmitted Power [dBm]")
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
    
    

    plt.title(f"Error vs Transmitted Power - {num_users} UEs")
    plt.xlabel("Transmitted Power [dBm]")
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




def plots_of_compare_SubSpaceNET_to_music_singal_band(results,input_power_list,num_users,model_path):
    plt.figure(figsize=(8, 5))
    plt.plot(input_power_list, results[0][0], marker='o', color='blue', label='Avg (with NN)')
    plt.plot(input_power_list, results[1][0], marker='o', color='orange', label='Avg (no NN)')
    
    plt.title(f"Error vs Transmitted Power - {num_users} UEs")
    plt.xlabel("Transmitted Power [dBm]")
    plt.ylabel("Error [m]")
    plt.yscale('log')
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

    plt.title(f"Error vs Transmitted Power - {num_users} UEs")
    plt.xlabel("Transmitted Power [dBm]")
    plt.ylabel("Error [m]")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # Save to same folder as model_path
    save_dir = os.path.dirname(model_path)
    plt.savefig(os.path.join(save_dir, f"Median_error_vs_input_power_{num_users}UEs.png"))
    plt.savefig(os.path.join("results", f"Median_error_vs_input_power_{num_users}UEs.png"))
    plt.close()

def plots_of_compare_MultiBandNet_to_MultiBeamformer(results,input_power_list,num_users,model_path):
    plt.figure(figsize=(8, 5))
    plt.plot(input_power_list, results[4][0], marker='o', color='yellow', label='Avg 24GHz (no NN)')
    plt.plot(input_power_list, results[3][0], marker='o', color='blue', label='Avg 18GHz (no NN)')
    plt.plot(input_power_list, results[2][0], marker='o', color='green', label='Avg 12GHz (no NN)')
    plt.plot(input_power_list, results[1][0], marker='o', color='orange', label='Avg 6GHz (no NN)')
    plt.plot(input_power_list, results[5][0], marker='o', color='dimgray', label='Avg MultiBeamformer(no NN)')
    plt.plot(input_power_list, results[0][0], marker='o', color='red', label='Avg MultiBandNet (with NN)')
    
    plt.title(f"Error vs Transmitted Power - {num_users} UEs")
    plt.xlabel("Transmitted Power [dBm]")
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
    plt.plot(input_power_list, results[5][1], marker='s', color='dimgray', label='Median MultiBeamformer(no NN)')
    plt.plot(input_power_list, results[0][1], marker='s', color='red', label='Median multiband (with NN)')
    
    

    plt.title(f"Error vs Transmitted Power - {num_users} UEs")
    plt.xlabel("Transmitted Power [dBm]")
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

