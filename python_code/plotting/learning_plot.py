import matplotlib.pyplot as plt
import os
from dir_definitions import ROOT_DIR
def plot_validation(error_list,save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(40,40*(len(error_list)+1),40),error_list)
    plt.title(f"Error vs batch")
    plt.xlabel("batch")
    plt.ylabel("Error [m]")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"training_learn_curve"))
    plt.savefig(os.path.join(ROOT_DIR,"results", f"training_learn_curve"))
    plt.close()
