import os, glob
import torch
import numpy as np 
import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import copy
import warnings
import logging
from scipy.io import loadmat
from scipy import misc
from PIL import Image
import imageio
import torchvision.transforms as transforms
from copy import deepcopy
import pdb
logger = logging.getLogger('global')
warnings.filterwarnings("ignore")
from utils.util import generate_mask
import tifffile as tiff
import nibabel
from datasets.transform import Composer, registered_workers 
from matplotlib import pyplot as plt
class PairedDataset(Dataset):
    def __init__(self, opt, train=True, augment=True):
        super(PairedDataset, self).__init__()
        self.train = train
        self.augment = augment
        self.opt = opt     
        # find images from folder/txt/name list 
        try:
            if self.train:
                self.img_path = opt.img_path
                self.label_path = opt.label_path
            else:
                self.img_path = opt.vimg_path
                self.label_path = opt.vlabel_path    
            self.img_ext = opt.img_ext
            self.label_ext = opt.label_ext  
            if os.path.isfile(self.img_path):
                with open(self.img_path) as f:
                    self.datalist = f.read().splitlines() 
            elif type(self.img_path) == list:
                self.datalist = self.img_path   
            elif type(self.img_path) == str:            
                self.datalist = sorted(list(Path(self.img_path).glob('*.'+self.img_ext)))
            # find labels from folder/txt/name list
            if os.path.isfile(self.label_path):
                with open(self.label_path) as f:
                    self.labellist = f.read().splitlines() 
            elif type(self.label_path) == list: 
                self.labellist = self.label_path 
            elif type(self.label_path) == str:                   
                self.labellist = sorted(list(Path(self.label_path).glob('*.'+self.label_ext)))
            # assert label and image are aligned
            assert len(self.datalist) == len(self.labellist)
        except:
            self.datalist, self.labellist = [], []
        # define transformation using opt      
        self.transform = self._get_transform()
    def name(self):
        return 'PairedDataset'
    def __getitem__(self, index):
        # load image
        self.image = self._get_image(index)
        # load label
        self.label = self._get_label(index)
        # add additional dimension for augmentation
        self.image = self.image[np.newaxis]
        self.label = self.label[np.newaxis]
        self.image, self.label = self.transform(self.image, self.label)
        # remove the additional dimension of label
        # image: [1, H, W], label: [H, W]
        sample = {'data':self.image, 'label':self.label[0], 'filename':str(self.datalist[index]),
                  'transform':self.transform.get_params()}
        return sample
    def __len__(self):
        return len(self.datalist)
    def _get_image(self,index):
        pass
    def _get_label(self,index):
        pass
    def _get_transform(self):
        pass



class OCTSegDataset(PairedDataset):
    # specifically for OCT segmentation dataset
    # https://github.com/YufanHe/oct_preprocess
    def __init__(self, opt, train=True, augment=True):
        super(OCTSegDataset, self).__init__(opt, train, augment)
    def _get_image(self,index):
        try:
            image = np.array(imageio.imread(str(self.datalist[index]),pilmode = 'L'))/255
        except:
            raise(RuntimeError("image type not supported"))
        return image
    def _get_label(self,index):
        with open(str(self.labellist[index]),'r') as f:
            dicts = json.loads(f.read())
        if 'lesion' in dicts.keys():
            mask = np.array(dicts['lesion'])
            mask[mask>1] = 1
        else:
            mask = np.zeros(self.image.shape)
        # add dtype=float to make sure None is converted to NaN
        bds = np.array(dicts['bds'], dtype=np.float) - 1 
        label = generate_mask(bd_pts=bds,lesion=mask)
        return label
    def _get_transform(self):
        # dataset specific augmentation
        transform_list = []
        if self.augment:
            transform_list +=[registered_workers['gamma'](\
                              {'p':self.opt.aug_prob,
                               'gamma':self.opt.gamma})]
            transform_list += [registered_workers['affine'](\
                               {'p':self.opt.aug_prob,
                                'angle':self.opt.affine_angle,
                                'translate':self.opt.affine_translate,
                                'scale':self.opt.affine_scale})]
            transform_list += [registered_workers['hflip']({'p':self.opt.aug_prob})]
            transform_list +=[registered_workers['noise'](\
                              {'p':self.opt.aug_prob,
                               'std':self.opt.noise_std})]
        transform_list += [registered_workers['normalize']({'n_label':False})]
        return Composer(transform_list)        
