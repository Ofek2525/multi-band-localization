import numpy as np
import torch

from exp_params import NS
from utils.basis_functions import compute_angle_options,compute_time_options
from dir_definitions import RAYTRACING_DIR
from channel.channel_loader import generate_batches, get_ues_info, generate_batches_by_rows

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_channel(ues_loc, band, input_power=None):
    """
    :param ues_loc: [ue_num_simultaneously X 2]
    :param band:
    :return: bs=bs_loc, y=y -> [butch X Nr X K X T], TOA=toas, AOA=aoas, band=band
    """
    band_freq_file_in_k = int(band.fc / 1000)
    csv_filename = rf"{RAYTRACING_DIR}/{band_freq_file_in_k}000/LOS_bs1_{band_freq_file_in_k}k.csv"

    ues_data = get_ues_info(csv_filename, ues_loc, input_power)
    y = sum([single_ue_channel(ue_data, band) for ue_data in ues_data]).unsqueeze(0)
    return y, ues_data


def random_pos_ues_channel(band, batch_size, ues_num, state="train"):
    """
    :param band:
    :param batch_size:
    :param ues_num:  number of ues simultaneously
    :return:ys,ues_data
    """

    ues_data = generate_batches(band, batch_size, ues_num, state)
    ys = []
    for i in range(batch_size):
        ys.append(sum(single_ue_channel(ues_data[i][ue],band) for ue in range(ues_num)))
    ys = torch.stack(ys, dim=0)
    return ys, ues_data


def ues_rows_channel(band, batch_size, ues_num, csv_rows_per_sample, state="train"):
    """
    :param band:
    :param batch_size:
    :param ues_num:  number of ues simultaneously
    :return:ys,ues_data
    """

    ues_data = generate_batches_by_rows(band, csv_rows_per_sample, state)
    ys = []
    for i in range(batch_size):
        ys.append(sum(single_ue_channel(ues_data[i][ue],band) for ue in range(ues_num)))
    ys = torch.stack(ys, dim=0)
    return ys, ues_data


def single_ue_channel(ue_data,band):
    """
    :param ue_data: dictionaire with keys: ue_loc, n_path, powers, toa, aoa
    :param band:
    :return:
    """
    NF = 7  # noise figure in dB
    N_0 = -174  # dBm
    toa_steering = torch.tensor(compute_time_options(band.fc, band.K, band.BW, values=np.array(ue_data['toa'])),device=DEVICE)
    aoa_steering = torch.tensor(compute_angle_options(np.sin(np.deg2rad(ue_data['aoa'])), values=np.arange(band.Nr)).T,device = DEVICE)
    powers = watt_power_from_dbm(torch.tensor(ue_data['powers'],device=DEVICE)).unsqueeze(0)
    aoa_steering *= powers
    h = aoa_steering.mm(toa_steering).unsqueeze(-1)
    BW_loss = 10 * np.log10(band.BW * 10 ** 6)
    noise_amp = watt_power_from_dbm(NF + BW_loss + N_0)
    S = torch.exp(1j * torch.rand((1, 1, NS), device=DEVICE) * 2 * np.pi)
    normal_gaussian_noise = 1 / np.sqrt(2) *(torch.randn((band.Nr, band.K, NS), device=DEVICE)+1j*torch.randn((band.Nr, band.K, NS), device=DEVICE))
    y= h*S/noise_amp+normal_gaussian_noise
    return y


def watt_power_from_dbm(dbm_power):
    return 10 ** ((dbm_power - 30) / 10)

