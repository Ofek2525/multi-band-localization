from numpy import linalg
from numpy import array

import torch
def too_close(locs,threshold):
    for i in range(len(locs)-1):
        for j in range(i+1,len(locs)):
            if linalg.norm(locs[i]-locs[j]) < threshold:
                return True
    return False        


if __name__ == "__main__":
    print(too_close(array([[-5000,0],[300,0],[0,9999],[0,400]]),499))
    K = 3
    N = 2
    RY = torch.arange(1,37).reshape((6,6))
    print(RY)
    reorder =RY.reshape((-1,K,N,K,N))
    reorder =reorder.permute(0,2,1,4,3).contiguous()
    reorder = reorder.reshape(-1,K*N,K*N)
    print(reorder)
    K = 2
    N = 3
    reorder =reorder.reshape((K,N,K,N))
    reorder =reorder.permute(1,0,3,2).contiguous()
    reorder = reorder.reshape(K*N,K*N)
    print(reorder)