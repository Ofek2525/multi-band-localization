import pandas as pd
import numpy as np
import time

from utils.bands_manipulation import Band, get_bands_from_conf
from exp_params import K, Nr, fc, BW, input_power
from dir_definitions import RAYTRACING_DIR, ALLBSs_DIR

'''
    לבדוק:
    1. את העניין של AOD AOA
'''

def sample_random_ues(csv_filename: str, ues_num: int):
    df = pd.read_csv(csv_filename)

    # Ensure ues_num does not exceed the number of available rows
    ues_num = min(ues_num, len(df))

    # Randomly sample ues_num rows
    sampled_df = df.sample(n=ues_num)

    results = []
    for _, row in sampled_df.iterrows():
        ue_loc = [row['rx_x'], row['rx_y']]
        n_path = int(row['n_path'])

        # Extract path-related values
        powers = [(input_power - row[f'path_loss_{i}']) for i in range(1, n_path + 1) if f'path_loss_{i}' in row]
        toa = [row[f'delay_{i}'] / (10 ** (-6)) for i in range(1, n_path + 1) if f'delay_{i}' in row]
        aoa = [row[f'aod_{i}'] + 90 for i in range(1, n_path + 1) if f'aod_{i}' in row]

        results.append({
            "ue_loc": ue_loc,
            "n_path": n_path,
            "powers": powers,
            "toa": toa,
            "aoa": aoa
        })

    return results


def get_ue_info_by_row(csv_filename: str, row_num: int,sweeped_power=None, rand_aoa_flag=False):
    df = pd.read_csv(csv_filename)

    if row_num >= len(df):
        return None  # Handle out-of-bounds safely
     
    row = df.iloc[row_num]
    n_path = int(row['n_path'])
    ue_power = input_power
    if sweeped_power is not None:
        ue_power = sweeped_power
    
    # Augmentation for the angle
    angle_change = 0
    if rand_aoa_flag:
        real_los_aoa = row['aod_1'] + 90
        new_los_aoa = np.random.uniform(-90, 90)
        angle_change = new_los_aoa - real_los_aoa
    
    powers = [(ue_power - row[f'path_loss_{i}']) for i in range(1, n_path + 1) if f'path_loss_{i}' in row]
    toa = [row[f'delay_{i}'] / 1e-6 for i in range(1, n_path + 1) if f'delay_{i}' in row]
    aoa = [row[f'aod_{i}'] + 90 + angle_change for i in range(1, n_path + 1) if f'aod_{i}' in row]

    return {
        "row_num": row_num,
        "bs_loc": [int(row['tx_x']), int(row['tx_y']), int(row['tx_z'])],
        "ue_loc": [int(row['rx_x']), int(row['rx_y']), int(row['rx_z'])],
        "n_path": n_path,
        "powers": powers,
        "toa": toa,
        "aoa": aoa
    }


def get_ues_info(csv_filename: str, ue_locs: list, sweeped_power=None):
    df = pd.read_csv(csv_filename)
    ue_power = input_power
    if sweeped_power is not None:
        ue_power = sweeped_power
    results = []
    for ue_loc in ue_locs:
        rx_x, rx_y = ue_loc

        # Find the matching row(s)
        matching_rows = df[(df['rx_x'] == rx_x) & (df['rx_y'] == rx_y)]

        for _, row in matching_rows.iterrows():
            n_path = int(row['n_path'])

            # Extract path-related values
            powers = [(ue_power - row[f'path_loss_{i}']) for i in range(1, n_path + 1) if f'path_loss_{i}' in row]
            toa = [row[f'delay_{i}'] / (10 ** (-6)) for i in range(1, n_path + 1) if f'delay_{i}' in row]
            aoa = [row[f'aod_{i}'] + 90 for i in range(1, n_path + 1) if f'aod_{i}' in row]

            results.append({
                "bs_loc": [int(row['tx_x']), int(row['tx_y']), int(row['tx_z'])],
                "ue_loc": [int(row['rx_x']), int(row['rx_y']), int(row['rx_z'])],
                "n_path": n_path,
                "powers": powers,
                "toa": toa,
                "aoa": aoa
            })

    return results



def generate_batches_by_rows(band: Band, csv_rows_per_sample, BS_num, state="train",input_power= None, augmentation=False):
    band_freq_file_in_G = int(band.fc / 1000)
    csv_filename = rf"{ALLBSs_DIR}/bs_{BS_num}/{state}_{band_freq_file_in_G}Ghz.csv"

    batch = []
    for sample in csv_rows_per_sample:
        results = []
        for ue_row in sample:
            results.append(get_ue_info_by_row(csv_filename, ue_row,input_power, rand_aoa_flag=augmentation))
        batch.append(results)
    return batch

