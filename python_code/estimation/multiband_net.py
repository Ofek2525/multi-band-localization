import torch
import torch.nn as nn
from exp_params import K, Nr

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Encoder 6k
class Encoder_6k(nn.Module):
    def __init__(self):
        super(Encoder_6k, self).__init__()
        self.two_orders = two_orders_block(Nr[0],K[0])     
        self.conv1 = nn.Conv2d(32, 16, kernel_size=2,stride=2)
        self.antirectifier = AntiRectifier()
       

    def forward(self, x: torch.Tensor):
        ''''''
        ''' pre processing '''
        x = torch.nn.functional.normalize(x, p=2, dim=(1,2), eps=1e-12)######
        x = x.unsqueeze(dim=1)  # Shape: [Batch size, 1, N, N]
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)  # Shape: [Batch size,2, N, N]
        x = x.type(torch.float) 
        ''' Architecture flow '''
        x = self.two_orders(x) #[Batch size,16, N, N]
        x = self.antirectifier(x) # Shape: [Batch size,32, N, N]
        
        # CNN block #1
        x_skip = self.conv1(x)  # Shape: [Batch size, 16, N/2, N/2]
        x = self.antirectifier(x_skip)  # Shape: [Batch size, 32, N/2, N/2]
        return x, x_skip


# Encoder 12k
class Encoder_12k(nn.Module):
    def __init__(self):
        super(Encoder_12k, self).__init__()
        self.two_orders = two_orders_block(Nr[1],K[1])

        self.conv1 = nn.Conv2d(32, 16, kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2,stride=2)
       
        self.antirectifier = AntiRectifier()
       

    def forward(self, x: torch.Tensor):
        ''''''
        ''' pre processing '''
        x = torch.nn.functional.normalize(x, p=2, dim=(1,2), eps=1e-12)######
        x = x.unsqueeze(dim=1)  # Shape: [Batch size, 1, N, N]
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)  # Shape: [Batch size,2, N, N]
        x = x.type(torch.float)
    
        ''' Architecture flow '''
        x = self.two_orders(x) #[Batch size,16, N, N]
        x = self.antirectifier(x) # Shape: [Batch size,32, N, N]
        
        # CNN block #1
        x = self.conv1(x)  # Shape: [Batch size, 16, N/2, N/2]
        x_skip = self.antirectifier(x)  # Shape: [Batch size, 32, N/2, N/2]
        # CNN block #2
        x = self.conv2(x_skip)  # Shape: [Batch size, 32, N/4, N/4]
        x = self.antirectifier(x)  # Shape: [Batch size, 64, N/4, N/4]
        return x, x_skip


# Encoder 18k
class Encoder_18k(nn.Module):
    def __init__(self):
        super(Encoder_18k, self).__init__()
        self.two_orders = two_orders_block(Nr[2],K[2])     

        self.conv1 = nn.Conv2d(32, 16, kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2,stride=2)

        self.antirectifier = AntiRectifier()

    def forward(self, x: torch.Tensor):
        ''''''
        ''' pre processing '''
        x = torch.nn.functional.normalize(x, p=2, dim=(1,2), eps=1e-12)######
        x = x.unsqueeze(dim=1)  # Shape: [Batch size, 1, N, N]
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)  # Shape: [Batch size,2, N, N]
        x = x.type(torch.float)
    
        ''' Architecture flow '''
        x = self.two_orders(x)# Shape: [Batch size,16, N, N]
        x = self.antirectifier(x) # Shape: [Batch size,32, N, N]
        
        # CNN block #1
        x = self.conv1(x)  # Shape: [Batch size, 32, N/2, N/2]
        x = self.antirectifier(x)  # Shape: [Batch size, 32, 
        # CNN block #2
        x = self.conv2(x)  # Shape: [Batch size, 32,
        x_skip = self.antirectifier(x)  # Shape: [Batch size, 64, 
        # CNN block #3
        x = self.conv3(x_skip)  # Shape: [Batch size, 64, 
        x = self.antirectifier(x)  # Shape: [Batch size, 128, N/8, N/8]
        return x, x_skip


# Encoder 24k
class Encoder_24k(nn.Module):
    def __init__(self):
        super(Encoder_24k, self).__init__()       
        self.two_orders = two_orders_block(Nr[3],K[3])

        self.conv1 = nn.Conv2d(32, 16, kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2,stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2,stride=2)

        self.antirectifier = AntiRectifier()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        ''''''
        ''' pre processing '''
        x = torch.nn.functional.normalize(x, p=2, dim=(1,2), eps=1e-12)######
        x = x.unsqueeze(dim=1)  # Shape: [Batch size, 1, N, N]
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)  # Shape: [Batch size,2, N, N]
        x = x.type(torch.float)
    
        ''' Architecture flow '''
        x = self.two_orders(x) # Shape: [Batch size,16, N, N]
        x = self.antirectifier(x) # Shape: [Batch size,32, N, N]
        
        # CNN block #1
        x = self.conv1(x)  # Shape: [Batch size, 32, N-1, N-1]
        x = self.antirectifier(x)  # Shape: [Batch size, 32, 2N-1, N-1]
        # CNN block #2
        x = self.conv2(x)  # Shape: [Batch size, 32, 2N-2, N-2]
        x = self.antirectifier(x)  # Shape: [Batch size, 64, 2N-2, N-2]
        # CNN block #3
        x = self.conv3(x)  # Shape: [Batch size, 64, 
        x_skip = self.antirectifier(x)  # Shape: [Batch size, 128, 

        x = self.conv4(x_skip)  # Shape: [Batch size, 128, N/16, N/16]
        x = self.antirectifier(x)  # Shape: [Batch size, 256, N/16, N/16]

        return x, x_skip


# Decoder
class Decoder(nn.Module):
    def __init__(self, Cin=480, psd_epsilon: float=0.1):
        super(Decoder, self).__init__()
        self.p = 0.3
        self.psd_epsilon = psd_epsilon
        self.deconv2 = nn.ConvTranspose2d(Cin, 128, kernel_size=2,stride=2)
        self.deconv3 = nn.ConvTranspose2d(256, 32, kernel_size=2,stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 2, kernel_size=2,stride=2)
        self.antirectifier = AntiRectifier()
        self.relu = nn.ReLU()
        self.DropOut = nn.Dropout(self.p)
    
    def forward(self, x: torch.Tensor, x_skip: torch.Tensor, batch_size, N):
        # DCNN block #2
        x = self.deconv2(x)  # Shape: [Batch size, 32, 2N-2, N-2]
        x = self.antirectifier(x)  # Shape: [Batch size, 64, 2N-2, N-2]
        # DCNN block #3
        x[:,:224,:,:] += x_skip
        x = self.deconv3(x)  # Shape: [Batch size, 16, 2N-1, N-1]
        x = self.antirectifier(x)  # Shape: [Batch size, 32, 2N-1, N-1]
        # DCNN block #4
        x = self.DropOut(x) #+x_skip
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
        Rz = (Kx_tag + torch.conj(torch.transpose(Kx_tag, 1,2)))/2
        # Feed surrogate covariance to the differentiable subspace algorithm
        Rz = torch.nn.functional.normalize(Rz, p=2, dim=(1,2), eps=1e-12) + I_N
        return Rz


# Multi Band SubSpaceNET
class Multi_Band_SubSpaceNET(nn.Module):
    def __init__(self, 
                 encoder_6k = None,
                 encoder_12k = None,
                 encoder_18k = None,
                 encoder_24k = None,
                 decoder = None):
        super(Multi_Band_SubSpaceNET, self).__init__()
        self.encoder_6k = encoder_6k if encoder_6k else Encoder_6k() 
        self.encoder_12k = encoder_12k if encoder_12k else Encoder_12k()
        self.encoder_18k = encoder_18k if encoder_18k else Encoder_18k()
        self.encoder_24k = encoder_24k if encoder_24k else Encoder_24k()
        self.decoder = decoder if decoder else Decoder()
        self.DropOut = nn.Dropout(0)

    def forward(self, x_list):
        x6, x12, x18, x24 = x_list
        batch_size,_, N = x18.shape

        # Encode each input separately
        x6_encoded, x6_skip = self.encoder_6k(x6)
        x12_encoded, x12_skip = self.encoder_12k(x12)
        x18_encoded, x18_skip = self.encoder_18k(x18)
        x24_encoded, x24_skip = self.encoder_24k(x24)

        # Concatenate along the channel dimension
        x_encoded = torch.cat([x6_encoded, x12_encoded, x18_encoded, x24_encoded], dim=1)
        x_encoded = self.DropOut(x_encoded)
        x_skip = torch.cat([x12_skip, x18_skip, x24_skip], dim=1)

        # Decode
        Rz = self.decoder(x_encoded, x_skip, batch_size, N)
        return Rz


# Init Encoders
class Init_Single_Band_SubSpaceNET(nn.Module):
    def __init__(self, encoder, decoder: Decoder):
        super(Init_Single_Band_SubSpaceNET, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor):
        batch_size, _, N = x.shape
        
        x_encoded, x_skip = self.encoder(x)
        Rz = self.decoder(x_encoded, x_skip, batch_size, N)
        return Rz

##########################################################################################
class AntiRectifier(nn.Module):
    def __init__(self, relu_inplace=False):
        super(AntiRectifier, self).__init__()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def forward(self, x):
        return torch.cat((self.relu(x), self.relu(-x)), 1)


class two_orders_block(nn.Module):
    def __init__(self, Nr,K,S = 4 ,p=0.3):
        super(two_orders_block, self).__init__()
        self.Nr = Nr 
        self.K = K
        self.yconv = nn.Conv2d(2, S, kernel_size=2,stride=1)
        self.xconv = nn.Conv2d(2, S, kernel_size=2,stride=1)

        self.ydeconv = nn.ConvTranspose2d(2*S, 2*S, kernel_size=2,stride=1)
        self.xdeconv = nn.ConvTranspose2d(2*S, 2*S, kernel_size=2,stride=1)
        self.DropOut = nn.Dropout(p)
        self.antirectifier = AntiRectifier()
        
    def forward(self, x):
        y = reorder(x,self.Nr,self.K) # Shape: [Batch size,2, N, N]
        y = self.yconv(y) # Shape: [Batch size,4, N-1, N-1]
        y = self.antirectifier(y) # Shape: [Batch size,8, N-1, N-1]
        y = self.ydeconv(y) # Shape: [Batch size,8, N, N]
        y = order_back(y,self.Nr,self.K)

        x = self.xconv(x) # Shape: [Batch size,4, N-1, N-1]
        x = self.antirectifier(x) # Shape: [Batch size,8, N-1, N-1]
        x = self.xdeconv(x) # Shape: [Batch size,8, N, N]

        x = torch.cat((x, y), dim=1)# Shape: [Batch size,16, N, N] or [Batch size,4*S, N, N]
        x = self.DropOut(x)
        return x


def reorder(x,nr,k):
    reorderd =x.reshape(x.shape[0],x.shape[1],k,nr,k,nr)
    reorderd =reorderd.permute(0,1,3,2,5,4).contiguous()
    return reorderd.reshape(x.shape[0],x.shape[1],k*nr,k*nr)


def order_back(x,nr,k):
    return reorder(x,k,nr)


