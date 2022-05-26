import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import copy
import pdb
from models.backends import UNet, ConvBlock
from utils.util import ncc, l2_reg_ortho
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import logging
import tifffile as tiff
from functools import partial
logger = logging.getLogger('global')

def init_weights(x):
    torch.manual_seed(0)
    if type(x) == nn.Conv2d:
        nn.init.kaiming_normal_(x.weight.data)
        nn.init.zeros_(x.bias.data)   

def init_weights_eye(x, channel=64):
    # indentity init, only works for same input/output channel
    if type(x) == nn.Conv2d:
        eye = nn.init.eye_(torch.empty(x.weight.shape[0], x.weight.shape[1])).unsqueeze(-1).unsqueeze(-1)
        init_bias = nn.init.zeros_(torch.empty(x.weight.shape[0]))            
        x.weight.data = eye
        x.bias.data = init_bias   

class DTTAnorm(nn.Module):
    def __init__(self):
        super(DTTAnorm,self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,16,3,padding=1)
        self.conv3 = nn.Conv2d(16,1,3,padding=1)
    def forward(self,x):
        x_ = self.conv1(x)
        scale = (torch.randn([1,16,1,1]) * 0.05 + 0.2).to(x_.device)
        x_ = torch.exp(-(x_**2) / (scale**2))
        x_ = self.conv2(x_)
        x_ = torch.exp(-(x_**2) / (scale**2))
        x_ = self.conv3(x_)
        return x_ + x

class ANet(nn.Module):
    def __init__(self, adpNet=None, tnet_dim=[1,64,64,64,64,1], seq=None):
        """Adaptor Net: Default is for a 4-level UNet with fixed 64 channels
        Args:
            AENet: nn.Module, pre-trained auto-encoder for the source image
            channel: int, input feature channel of the affine transform
            seq: list->int, index of the affine matrix to be used 
        """
        super(ANet,self).__init__()
        self.conv = nn.ModuleList()
        feature_channel = tnet_dim[1:-1]
        self.channel = feature_channel + feature_channel[::-1]
        nums = len(self.channel)
        self.nums = nums
        if seq is None:
            self.seq = np.arange(nums)
        else:
            self.seq = seq
        # use pre-contrast manipulation 
        self.adpNet = adpNet
        if adpNet is None:
            self.adpNet = nn.Sequential(
                    nn.Conv2d(1,64,1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.InstanceNorm2d(64),
                    nn.Conv2d(64,64,1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.InstanceNorm2d(64),   
                    nn.Conv2d(64,1,1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.InstanceNorm2d(1)                    
                )
            self.adpNet.apply(init_weights)     
        # use feature affine transform
        for c in self.channel:
            convs = nn.Conv2d(c,c,1)
            self.conv.append(convs)
        self.conv.apply(init_weights_eye)
    def reset(self):
        # reset the fine-tuned weights for a new test subject
        np.random.seed(0)
        torch.manual_seed(0)
        self.conv.apply(init_weights_eye)
        self.adpNet.apply(init_weights)
        self.cuda()
    def forward(self, x, TNet, AENet, side_out=False):
        """
        Forward for a 4-level UNet
        Args: 
            TNet: nn.Module. The pretrained task network
            side_out: bool. If true, output every intermediate results
            seq: list->int or np array. Position of 1x1 convolution
        """
        x = self.adpNet(x)
        xh = [x]      
        x = TNet.inblocks(x)
        ct = 0
        # apply 1x1 conv on input blocks
        seq = self.seq
        if ct in seq:
            x = self.conv[ct](x)
        ct += 1
        xh.append(x)
        for i in range(TNet.depth):
            x = TNet.downblocks[i](x)    
            # apply 1x1 conv on every downsample output except bottleneck       
            if ct in seq:
                x = self.conv[ct](x)
            ct += 1            
            xh.append(x)                    
        x = TNet.bottleneck(x)
        if ct in seq:
            x = self.conv[ct](x)
        ct += 1          
        xh.append(x)
        for i in range(TNet.depth):
            x = TNet.upblocks[TNet.depth-i-1](x,xh[TNet.depth-i])
            if ct in seq:
                x = self.conv[ct](x)
            ct += 1            
            xh.append(x)        
        x = TNet.outblock(x)
        xh.append(x)
        if side_out:
            return xh
        else:
            return x 