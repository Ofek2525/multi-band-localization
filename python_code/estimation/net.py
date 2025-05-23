import torch
import torch.nn as nn
from exp_params import K, Nr, main_band_idx
#from python_code import DEVICE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SubSpaceNET(nn.Module):
    def __init__(self, psd_epsilon: float=0.1):
        super(SubSpaceNET, self).__init__()
        self.p1 = 0.2
        self.p2 = 0.2
        self.psd_epsilon = psd_epsilon

        self.yconv = nn.Conv2d(2, 4, kernel_size=2,stride=1)
        self.xconv = nn.Conv2d(2, 4, kernel_size=2,stride=1)

        self.ydeconv = nn.ConvTranspose2d(8, 8, kernel_size=2,stride=1)
        self.xdeconv = nn.ConvTranspose2d(8, 8, kernel_size=2,stride=1)        

        self.conv1 = nn.Conv2d(32, 16, kernel_size=2,stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2,stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2,stride=1)
        
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2,stride=1)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2,stride=1)
        self.deconv4 = nn.ConvTranspose2d(32, 2, kernel_size=2,stride=1)
        self.antirectifier = AntiRectifier()
        self.relu = nn.ReLU()
        self.DropOut1 = nn.Dropout(self.p1)
        self.DropOut2 = nn.Dropout(self.p2)

    def forward(self, x: torch.Tensor):
        ''''''
        ''' pre processing '''
        x = torch.nn.functional.normalize(x, p=2, dim=(1,2), eps=1e-12)######
        x = x.unsqueeze(dim=1)  # Shape: [Batch size, 1, N, N]
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)  # Shape: [Batch size,2, N, N]
        batch_size,_, _, N = x.shape
        x = x.type(torch.float)
    
        ''' Architecture flow '''
        y = reorder(x,Nr[0],K[0]) # Shape: [Batch size,2, N, N]
        y = self.yconv(y) # Shape: [Batch size,4, N-1, N-1]
        y = self.antirectifier(y) # Shape: [Batch size,8, N-1, N-1]
        y = self.ydeconv(y) # Shape: [Batch size,8, N, N]
        y = order_back(y,Nr[0],K[0])

        x = self.xconv(x) # Shape: [Batch size,4, N-1, N-1]
        x = self.antirectifier(x) # Shape: [Batch size,8, N-1, N-1]
        x = self.xdeconv(x) # Shape: [Batch size,8, N, N]

        x = torch.cat((x, y), dim=1)# Shape: [Batch size,16, N, N]
        x = self.DropOut1(x)
        x = self.antirectifier(x) # Shape: [Batch size,32, N, N]
        
        # CNN block #1
        x = self.conv1(x)  # Shape: [Batch size, 16, N-1, N-1]
        x1 = self.antirectifier(x)  # Shape: [Batch size, 32, 2N-1, N-1]
        # CNN block #2
        x = self.conv2(x1)  # Shape: [Batch size, 32, 2N-2, N-2]
        x = self.antirectifier(x)  # Shape: [Batch size, 64, 2N-2, N-2]
        # CNN block #3
        x = self.conv3(x)  # Shape: [Batch size, 64, 2N-3, N-3]
        x = self.antirectifier(x)  # Shape: [Batch size, 128, 2N-3, N-3]

       
        x = self.deconv2(x)  # Shape: [Batch size, 32, 2N-2, N-2]
        x = self.antirectifier(x)  # Shape: [Batch size, 64, 2N-2, N-2]
        # DCNN block #3
        x = self.deconv3(x)  # Shape: [Batch size, 16, 2N-1, N-1]
        x = self.antirectifier(x)  # Shape: [Batch size, 32, 2N-1, N-1]
        # DCNN block #4
        x = self.DropOut2(x + x1)
        x = self.deconv4(x) #+ x0  # Shape: [Batch size, 2, 2N, N]  + x0[:, 0].unsqueeze(1)

        # Reshape Output shape: [Batch size, 2N, N]
        # Real and Imaginary Reconstruction
        Rx_real = x[:,0, :, :]  # Shape: [Batch size, N, N])
        Rx_imag = x[:,1, :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag).to(torch.complex128)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        I_N = torch.eye(N)
        I_N = I_N.reshape((1, N, N))
        I_N = I_N.repeat(batch_size, 1, 1).to(DEVICE)
        #Rz = (torch.bmm(Kx_tag, torch.conj(torch.transpose(Kx_tag, 1,2))) + self.psd_epsilon * I_N)
        Rz = (Kx_tag + torch.conj(torch.transpose(Kx_tag, 1,2)))/2 + self.psd_epsilon * I_N
        # Feed surrogate covariance to the differentiable subspace algorithm
        Rz = torch.nn.functional.normalize(Rz, p=2, dim=(1,2), eps=1e-12)
        return Rz


class AntiRectifier(nn.Module):
    def __init__(self, relu_inplace=False):
        super(AntiRectifier, self).__init__()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def forward(self, x):
        return torch.cat((self.relu(x), self.relu(-x)), 1)


class single_nurone(nn.Module):
    def __init__(self):
        super(single_nurone, self).__init__()

    def forward(self, R):
        return R

def reorder(x,nr,k):
    reorderd =x.reshape(x.shape[0],x.shape[1],k,nr,k,nr)
    reorderd =reorderd.permute(0,1,3,2,5,4).contiguous()
    return reorderd.reshape(x.shape[0],x.shape[1],k*nr,k*nr)

def order_back(x,nr,k):
    return reorder(x,k,nr)