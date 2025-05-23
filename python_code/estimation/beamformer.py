import numpy as np
import torch

from exp_params import alg, aoa_res, T_res, plot_estimation_results
from utils.basis_functions import grid_basis_func
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def single_band_beamformer(band, RY):
    '''

    :param RY:
    :param band:
    :return:
    '''
    aoa_basis, toa_basis, aoa_grid, times_grid = grid_basis_func(band, T_res, aoa_res)
    aoa_basis = torch.tensor(aoa_basis).to(DEVICE)
    toa_basis = torch.tensor(toa_basis).to(DEVICE)
    W_basis = torch.einsum("im,pq->ipqm", aoa_basis, toa_basis).reshape(aoa_basis.shape[0], toa_basis.shape[0], -1)
    left_mul = torch.einsum("bij,mkj->bmki",RY,W_basis)
    spec = torch.real(torch.einsum("mkj,bmkj->bmk",W_basis.conj(),left_mul))
    return spec, aoa_grid, times_grid


def multi_band_beamformer(bands, per_band_RY):
    #from tomer's code
    ALG_THRESHOLD = 1.2
    K = len(per_band_RY)
    peak, chosen_k = None, None
    aoa_grid, times_grid = None,None
    norm_values_list = []
    for k in range(K):
        # compute the spectrum values for sub-band
        norm_values, a_grid, t_grid = single_band_beamformer(bands[k], per_band_RY[k])
        norm_values = np.sqrt(np.array(norm_values[0].to("cpu")))  
        maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
        norm_values_list.append(norm_values)
        # only if peak is above noise level
        if norm_values[maximum_ind[0], maximum_ind[1]] > ALG_THRESHOLD * np.mean(norm_values):
            aoa_grid, times_grid = a_grid, t_grid
            peak = maximum_ind
            chosen_k = k
    # if all are below the noise level - choose the last sub-band
    if chosen_k is None:
        aoa_grid, times_grid = a_grid, t_grid
        peak = maximum_ind
        chosen_k = K - 1

    return np.array([[[np.degrees(aoa_grid[peak[0]]),times_grid[peak[1]]]]]), torch.tensor(norm_values_list[chosen_k][np.newaxis,:,:]).to(DEVICE), aoa_grid, times_grid    
