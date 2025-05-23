import numpy as np
import matplotlib.pyplot as plt


def plot_angle_time(spectrum, aoa, toa, ues_pos,point_toa, point_aoa_deg,plot_path = r"results/AOA_and_delay_est.png",zoom = False):
    fig = plt.figure()
    plt.contourf(toa, np.degrees(aoa), np.log10(spectrum), cmap='magma')#
    ax = plt.gca()
    idx_aoas = [np.argmin(np.abs(aoa - np.deg2rad(p))) for p in point_aoa_deg]
    idx_toas = [np.argmin(np.abs(toa - p)) for p in point_toa]   
    for i in range(len(point_toa)):
        ax.scatter(point_aoa_deg[i],point_toa[i], marker="x", color='white',s=1, zorder=5) 
   
    #ax.legend(fontsize='small')
    ax.set_xlabel('TOA [us]')
    ax.set_ylabel('AOA [deg]')
    ax.set_title(f"Users locations: {ues_pos}")
    plt.savefig(plot_path, dpi=fig.dpi)
    plt.close(fig)
    if zoom:
        for i in range(len(point_toa)):
            fig = plt.figure()
            zoomed_toa = toa[max(0,idx_toas[i]-len(toa)//20):min(len(toa),idx_toas[i]+len(toa)//20+1)]
            zoomed_aoa = np.degrees(aoa[max(0,idx_aoas[i]-len(aoa)//20):min(len(aoa),idx_aoas[i]+len(aoa)//20+1)])
            plt.contourf(zoomed_toa, zoomed_aoa,
                          np.log10(spectrum[max(0,idx_aoas[i]-len(aoa)//20):min(len(aoa),idx_aoas[i]+len(aoa)//20+1),max(0,idx_toas[i]-len(toa)//20):min(len(toa),idx_toas[i]+len(toa)//20+1)]), cmap='magma')#
            ax = plt.gca()
            #ax.scatter(point_aoa_deg[i],point_toa[i], marker="x", color='white',s=1, zorder=5) 
            #ax.legend(fontsize='small')
            ax.set_xlabel('TOA [us]')
            ax.set_ylabel('AOA [deg]')
            ax.set_title(f"Users locations: {ues_pos}")
            plt.savefig(fr"results/zoom{i}", dpi=fig.dpi)
            plt.close(fig)
