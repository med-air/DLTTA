from email.mime import base
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import h5py
import scipy.io as scio
from glob import glob
import SimpleITK as sitk
import random
import cv2
import torchio as tio



def _label_decomp(label_vol, num_cls):
    """
    decompose label for softmax classifier
    original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    numpy version of tf.one_hot
    """
    one_hot = []
    for i in range(num_cls):
        _vol = np.zeros(label_vol.shape)
        _vol[label_vol == i] = 1
        one_hot.append(_vol)

    return np.stack(one_hot, axis=0)



class Normalization(object):
    
    
    def __init__(self):
        self.name = 'Normalization'

    def __call__(self, sample):
        resacleFilter = sitk.RescaleIntensityImageFilter()
        resacleFilter.SetOutputMaximum(255)
        resacleFilter.SetOutputMinimum(0)
        image, label = sample['image'], sample['label']
        image = resacleFilter.Execute(image)

        return {'image':image, 'label':label}

class Prostate(Dataset):
    '''
    Six prostate dataset (BIDMC, HK, I2CVB, ISBI, ISBI_1.5, UCL)
    '''
    def __init__(self, site, base_path=None):
        channels = {'BIDMC':3, 'HK':3, 'I2CVB':3, 'ISBI':3, 'ISBI_1.5':3, 'UCL':3}
        assert site in list(channels.keys())
       
        base_path = '/research/pheng4/qdliu/hzyang/prostate/test/'
        self.site = site
        self.base_path = base_path
        self.f_names = os.listdir(os.path.join(base_path, self.site))

       

    def __len__(self):
        return len(self.f_names)

    def __getitem__(self, idx):
        f_name = self.f_names[idx]
        #print(idx)
        sampledir = os.path.join(self.base_path, self.site, f_name)
        
        test_patch = np.load(sampledir)
        image_np = test_patch[0]
        label_np = test_patch[1]

        
        image = torch.Tensor(image_np)
        label = torch.Tensor(label_np)
        image = torch.unsqueeze(image, dim=0)
        label = torch.unsqueeze(label, dim=0)
        
       
        label = _label_decomp(label[0], 2)
        return image, label


if __name__=='__main__':
    pass