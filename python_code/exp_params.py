
# Bands parameters
K = [20,20,20,20]#[20]  # [50, 100, 75, 100]  # number of subcarriers
Nr = [4,8,16,32]#[8]  # [4, 8, 16, 24]  # number of elements in ULA
fc =[6000, 12000, 18000, 24000] #[24000]  # [6000, 12000, 18000, 24000]  # carrier frequency in MHz
BW = [4,4,4,4]#[12/5]  # [6, 12, 24, 48]  # BW frequency in MHz
main_band_idx = 2

input_power = 5 #3 dBm
NS = 50
num_of_BSs = 15

# parameters
alg = 'MUSIC'  # 'Beamformer','MUSIC','MultiBeamformer'
aoa_res = 0.2  # resolution in degrees for the azimuth dictionary
T_res = 0.03  # resolution in micro second for the delay dictionary

# general
seed = 1787  #1 run seed
plot_estimation_results = True  # whether to plot the estimation spectrum - True or False