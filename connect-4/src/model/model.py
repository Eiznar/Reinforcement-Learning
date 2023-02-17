
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from .blocks import ConvBlock, ResBlock, OutBlock

class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return np.int64(self.X[idx].transpose(2,0,1)), self.y_p[idx], self.y_v[idx]


class ConnectNet(nn.Module):
    def __init__(self, n_resblocks=19):
        super(ConnectNet, self).__init__()
        self.conv = ConvBlock()
        self.n_resblocks = n_resblocks
        for block in range(n_resblocks):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(self.n_resblocks):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

class AlphaLoss(torch.nn.Module):
    """Loss function described in the paper. It is defined as the sum of the MSE on value and cross-entropy policy losses.

    Args:
        torch (_type_): _description_
    """
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy* 
                                (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error