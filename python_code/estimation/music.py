import numpy as np
import torch

from exp_params import alg, aoa_res, T_res, plot_estimation_results
from utils.basis_functions import grid_basis_func
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






