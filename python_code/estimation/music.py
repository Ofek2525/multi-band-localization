import numpy as np
import torch

from exp_params import alg, aoa_res, T_res, plot_estimation_results,increase_res_factor
from utils.basis_functions import grid_basis_func,compute_angle_options,compute_time_options
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def music(band, RY, M):
    '''

    :param RY:
    :param band:
    :return:
    '''
    aoa_basis, toa_basis, aoa_grid, times_grid = grid_basis_func(band, T_res, aoa_res)
    eig_val, eig_vec = torch.linalg.eigh(RY)
    sorted_idx = torch.argsort(torch.real(eig_val))
    sorted_eigvectors = torch.gather(eig_vec, 2,sorted_idx.unsqueeze(-1).expand(-1, -1, RY.shape[-1]).transpose(1, 2))
    #eig_vec = eig_vec[:, :, torch.argsort(eig_val, dim=1)]
    U = sorted_eigvectors[:, :, :-M]
    U_H = U.conj().transpose(1,2)
    aoa_basis = torch.tensor(aoa_basis).to(DEVICE)
    toa_basis = torch.tensor(toa_basis).to(DEVICE)
    W_basis = torch.einsum("im,pq->ipqm", aoa_basis, toa_basis).reshape(aoa_basis.shape[0], toa_basis.shape[0], -1)
    music = 1 / (torch.norm(torch.einsum("bij,mkj->bmki",U_H,W_basis), dim=3)) ** 2
    #print(f"eig vals ={eig_val[0,-5:]}")
    return music, aoa_grid, times_grid


def increase_res(band, RY, aoa,toa):
    eig_val, eig_vec = torch.linalg.eigh(RY)
    sorted_idx = torch.argsort(torch.real(eig_val))
    sorted_eigvectors = torch.gather(eig_vec, 2,sorted_idx.unsqueeze(-1).expand(-1, -1, RY.shape[-1]).transpose(1, 2))
    U = sorted_eigvectors[:, :, :-aoa.shape[0]]
    U_H = U.conj().transpose(1,2)

    improved_aoa = np.zeros_like(aoa)
    improved_toa = np.zeros_like(toa)
    for ue_indx in range(aoa.shape[0]):
        aoa_grid = np.arange(max(-np.pi / 2,aoa[ue_indx]-aoa_res * np.pi / 180), min(np.pi / 2,aoa[ue_indx]+aoa_res * np.pi / 180), (1/increase_res_factor)*aoa_res * np.pi / 180)
        aoa_basis = compute_angle_options(np.sin(aoa_grid), values=np.arange(band.Nr))
        times_grid = np.arange(max(0,toa[ue_indx]-T_res), min(0.8*band.K / band.BW, toa[ue_indx]+T_res), T_res/increase_res_factor)
        toa_basis = compute_time_options(0, band.K, band.BW, values=times_grid,remove_duplicates = 0)
        aoa_basis = torch.tensor(aoa_basis).to(DEVICE)
        toa_basis = torch.tensor(toa_basis).to(DEVICE)
        W_basis = torch.einsum("im,pq->ipqm", aoa_basis, toa_basis).reshape(aoa_basis.shape[0], toa_basis.shape[0], -1)
        music = 1 / (torch.norm(torch.einsum("bij,mkj->bmki",U_H,W_basis), dim=3)) ** 2
        music = music[0] 
        music = music.cpu().numpy()
        maximum_ind = np.array(np.unravel_index(np.argmax(music, axis=None), music.shape))
        improved_aoa[ue_indx] = aoa_grid[maximum_ind[0]] 
        improved_toa[ue_indx] = times_grid[maximum_ind[1]]
        
    return improved_aoa,improved_toa       




