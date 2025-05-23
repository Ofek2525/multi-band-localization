import numpy as np
import torch

from exp_params import seed, K, Nr, fc, BW
from utils.basis_functions import compute_angle_options,compute_time_options
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def music_loss(RY, data_label, band, M):
    # data_label [bands x butch x M X {}]
    #D = 1
    data_label = data_label[0]
    mloss = 0
    eig_val, eig_vec = torch.linalg.eigh(RY)
    sorted_idx = torch.argsort(torch.real(eig_val))
    sorted_eigvectors = torch.gather(eig_vec, 2,sorted_idx.unsqueeze(-1).expand(-1, -1, RY.shape[-1]).transpose(1, 2))
    #sort!
    U = sorted_eigvectors[:, :, :-M] ###################################
    U_H = U.conj().transpose(1, 2)

    for b in range(len(data_label)):
        aoas = [data["aoa"][0] for data in data_label[b]]
        toas = [data["toa"][0] for data in data_label[b]]
        aoa_basis = compute_angle_options(np.sin(np.deg2rad(np.array(aoas))), values=np.arange(band.Nr)).T
        toa_basis = compute_time_options(0, band.K, band.BW, values=np.array(toas)).T
        aoa_basis = torch.tensor(aoa_basis).to(DEVICE)
        toa_basis = torch.tensor(toa_basis).to(DEVICE)
        W = (aoa_basis.unsqueeze(0) * toa_basis.unsqueeze(1)).reshape(-1, toa_basis.shape[1])
        R = U_H[b] @ W  # Shape: (r, M)
        # loss_list = torch.zeros((D))
        # loss_list[0]= torch.sum(torch.abs(R) ** 2)
        # for i in range(1,D):
        #     u_H = torch.cat((eig_vec[b,:,:-(i+1)],eig_vec[b,:,-i:]),dim=1).conj().transpose(0, 1)
        #     R = u_H @ W
        #     loss_list[i] = (torch.sum(torch.abs(R) ** 2))    
        mloss += torch.sum(torch.abs(R) ** 2) #+ 3*(torch.exp(3*torch.sum(eig_val[b,:-M])/((eig_val[b,-M]*eig_val.shape[1]))) - 3*torch.sum(eig_val[b,:-M])/((eig_val[b,-M]*eig_val.shape[1])) -1)

    return mloss/len(data_label)

