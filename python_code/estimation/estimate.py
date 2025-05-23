import torch
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.ndimage import label
from scipy.ndimage import find_objects

from estimation.music import music
from estimation.beamformer import single_band_beamformer,multi_band_beamformer
from estimation.net import single_nurone
from utils.bands_manipulation import get_bands_from_conf
from exp_params import seed, K, Nr, fc, BW, alg, aoa_res, T_res, plot_estimation_results, main_band_idx
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_k_highest_peaks(matrix, k):
    """
    Find the k highest peaks in a 2D matrix using SciPy tools. A peak is defined as a
    local maximum surrounded by smaller values.

    Parameters:
    - matrix (2D array-like): Input matrix.
    - k (int): Number of highest peaks to extract.

    Returns:
    - peaks (list): List of tuples (row, col, value) representing the positions and values of the k highest peaks.
    """
    # Apply maximum filter to find local maxima
    matrix = np.array(matrix.to("cpu"))
    neighborhood = maximum_filter(matrix, size=5, mode='constant', cval=-np.inf)
    local_max = (matrix == neighborhood)

    # Label the connected components of local maxima
    labeled, num_features = label(local_max)
    slices = find_objects(labeled)

    # Extract peak positions and values
    peaks = []
    try:
        for sl in slices:
            row = int((sl[0].start + sl[0].stop - 1) / 2)
            col = int((sl[1].start + sl[1].stop - 1) / 2)
            value = matrix[row, col]
            peaks.append([row, col, value])
    except Exception as e:
        pass

    # Sort peaks by value (descending) and select the top k
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)[:k]
    if len(peaks) < k:
        print(f"find_k_highest_peaks: Less than {k} peaks found.")
        # add random peaks
        x_random = np.random.randint(0, matrix.shape[0], (k - len(peaks),))
        y_random = np.random.randint(0, matrix.shape[1], (k - len(peaks),))
        for i in range(k - len(peaks)):
            peaks.append([x_random[i], y_random[i], matrix[x_random[i], y_random[i]]])
    return np.array(peaks)


def single_band_autocorrection(y):
    #y = torch.tensor(y).to(DEVICE)
    y = torch.permute(y, (0, 3, 2, 1)).contiguous()  # permute
    y = torch.reshape(y, (y.shape[0], y.shape[1], -1, 1))
    RY = torch.mean(torch.einsum("ijkm,ijml->ijkl",y, y.conj().transpose(2, 3)), dim=1)
    return RY


def estimate_evaluation(alg, multiband, per_band_y, bands, model, num_of_ues):
    '''
    estimate for evaluation

    :param alg: 'music' or 'beamforming'...
    :param multiband: 'MULTI' or 'SINGLE'
    :param per_band_y: list of y's butches
    :param bands: exp params

    :return: AOAs and TOAs estimation, with 'alg' algorithm [butch X ue_num_simultaneously], [butch X ue_num_simultaneously]
            , spectrum (for plotting)
    '''
    if alg == 'MultiBeamformer':
        assert num_of_ues == 1
        assert per_band_y[0].shape[0] == 1
        per_band_RY = []
        for y in per_band_y:
            per_band_RY.append(single_band_autocorrection(y))
        return multi_band_beamformer(bands, per_band_RY)    



    if multiband == 'SINGLE':
        RY = single_band_autocorrection(per_band_y[0])
        alternative_RY = model(RY)
        main_band = bands[0]
    elif multiband == 'MULTI':
        main_band = bands[main_band_idx]
        per_band_RY = []
        for y in per_band_y:
            per_band_RY.append(single_band_autocorrection(y))
        alternative_RY = model(per_band_RY)


    if alg == 'MUSIC':
        spec, aoa_grid, times_grid = music(main_band, alternative_RY, num_of_ues)
    if alg == 'Beamformer':
        spec, aoa_grid, times_grid =single_band_beamformer(main_band, alternative_RY)

    peaks = np.zeros((spec.shape[0], num_of_ues, 2))
    for idx, sample in enumerate(spec):
        peaks_idx = find_k_highest_peaks(sample, num_of_ues)
        aoa = aoa_grid[peaks_idx[:,0].astype(int)]
        toa = times_grid[peaks_idx[:,1].astype(int)]
        peaks[idx, :, 0] = np.degrees(aoa)
        peaks[idx, :, 1] = toa
        

    return peaks, spec,aoa_grid,times_grid

