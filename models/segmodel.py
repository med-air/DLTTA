import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb
from models.backends import UNet, ConvBlock
from utils.util import ncc, grams
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import logging
logger = logging.getLogger('global')
from models.basemodel import tonp, AdaptorNet
from functools import partial



def configure_model(model):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
    return model



class SegANet(AdaptorNet):
    def __init__(self, opt):
         super(SegANet,self).__init__(opt)      
    def def_loss(self):
        if self.opt.segloss == 'ce':
            self.TLoss = self.ce_loss
        elif self.opt.segloss == 'dice':
            self.TLoss = partial(self.dice_loss, softmax=True)
        self.AELoss = nn.MSELoss()
        if self.opt.segaeloss == 'mse':
            self.AELoss_out = nn.MSELoss()
        elif self.opt.segaeloss == 'dice':
            self.AELoss_out = self.dice_loss
    def ce_loss(self, pred, label):
        if pred.shape != label.shape:
            return F.cross_entropy(pred, label)
        else:
            return F.cross_entropy(pred, torch.argmax(label, 1))
    def dice_loss(self, pred, label, softmax=False):
        """Calculate dice loss:
        Args:
            pred: [batch, channel, img_row,img_col]
            label:[batch, img_row, img_col] or the same size with pred
        """
        if softmax:
            pred = F.softmax(pred,1)
        if pred.shape == label.shape:
            label_ = label
        else:
            label_ = pred.clone().detach().zero_()
            label_.scatter_(1, label.unsqueeze(1), 1)
        dice = (2*torch.sum(torch.sum(pred*label_,-1),-1)+0.001)/ \
        (torch.sum(torch.sum(pred,-1),-1) + torch.sum(torch.sum(label_,-1),-1) + 0.001)
        return 1 - torch.mean(dice)

    def cal_metric3d(self, pred, label, ignore=[0,9,10], fg=False):
        """Calculate quantitative metric: Dice coefficient
        Args: 
            pred: [ slice, channel, img_row, img_col]
            label: [ slice, img_row, img_col]
            ignore: ignore labels
            fg: return foreground dice
        Return:
            dice: list of dice coefficients
        """  
        if torch.is_tensor(pred):
            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()
        C = pred.shape[1]
        pred = np.argmax(pred, axis=1)
        if fg:
            C = 2
            pred[pred>0] = 1
        dice = np.zeros(C)
        for i in range(C):
            pl = pred == i
            ll = label == i
            sump = np.nansum(pl)
            sumt = np.nansum(ll)
            inter = np.nansum(pl*ll)
            if sumt == 0:
                dice[i] = np.nan
            else:
                dice[i] = 2*inter/(sump+sumt)   
        dice = np.delete(dice,ignore)    
        return dice
    def cal_metric(self, pred, label, ignore=[0,9,10]):
        """Calculate quantitative metric: Dice coefficient
        Args: 
            pred: [batch, channel, img_row, img_col]
            label: [batch, img_row, img_col]
            ignore: ignore labels
        Return:
            dice: list of dice coefficients, [batch]
        """  
        if torch.is_tensor(pred):
            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()
        dice = []
        batch_size, C = pred.shape[:2]
        pred = np.argmax(pred,1)
        for b_id in range(batch_size):
            _dice = np.zeros(C)
            for i in range(C):
                pl = pred[b_id,:,:] == i
                ll = label[b_id,:,:] == i
                sump = np.nansum(pl)
                sumt = np.nansum(ll)
                inter = np.nansum(pl*ll)
                if sumt == 0:
                    _dice[i] = np.nan
                else:
                    _dice[i] = 2*inter/(sump+sumt)  
            _dice = np.delete(_dice,ignore)
            dice.append(_dice)    
        return dice
    def test(self, return_pred=False):
        """Test using ANet and TNet
        """
        
        
        self.TNet.train()
        self.ANet.train()
        with torch.no_grad():
            pred = self.ANet(self.image, self.TNet, self.AENet[0], side_out=True)
            pred, adapt_img = pred[-1], pred[0]
            pred = F.softmax(pred, dim=1)
                
            
        metric = [[],[]]            
        if self.opt.__dict__.get('cal_metric',True):
            metric[0] = self.cal_metric(pred, self.label.squeeze(1))

        if return_pred:
            return metric, pred
        else:
            return metric
