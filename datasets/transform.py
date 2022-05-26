import numpy as np 
import cv2
import pdb
import logging
logger = logging.getLogger('global')
from matplotlib import pyplot as plt
from scipy import interpolate
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from monai.transforms import Affine, AdjustContrast, ResizeWithPadOrCrop
import random

class Composer(object):
    def __init__(self, workers):
        self.workers = workers
    def __call__(self, image, label):
        for _ in self.workers:
            image,label = _(image,label)
        return image, label
    def get_params(self):
        # get the latest transformation parameters
        # change after the composer being called
        params = {}
        for _ in self.workers:
            params[_.name()] = _.params
        return params
    def get_rdparams(self):
        # get the transformation random range
        rdparams = {}
        for _ in self.workers:
            rdparams[_.name()] = _.rdparams
        return rdparams

class Worker(object):
    def __init__(self, rdparams={}):
        pass
    def name(self):
        return 'Base class for transform'
    def _random(self):
        # generate final transformation params from rdparams
        pass
    def __call__(self, image, label, params=None):
        pass

class _normalize_worker(Worker):
    def __init__(self,rdparams={}):
        super(_normalize_worker,self).__init__(rdparams)
        default_rdparams = {'n_label':False,'mean':0.0,'std':1.0}
        default_rdparams.update(rdparams)
        self.rdparams = default_rdparams
    def name(self):
        return 'normalization'
    def __call__(self, image, label, params=None):
        if params is None:
            params = self._random()
        self.params = params
        # convert back to numpy and normalize to [0,1]
        image = image.astype(np.float32)
        image = (image - np.mean(image)) / np.std(image) * self.params['std'] + self.params['mean']
        if self.params['n_label']:
            label =  (label - np.mean(label)) / np.std(label)
            label = label.astype(np.float32)
        else:
            label = label.astype(np.uint8)
        return image, label   
    def _random(self):
        params = self.rdparams
        return params

class _affine_worker(Worker):
    def __init__(self,rdparams={}):
        super(_affine_worker,self).__init__(rdparams)
        default_rdparams = {'p':0.5,  'angle':[-30,30], 'label_mode':'nearest',
                            'translate':[-10,10,-10,10],
                            'scale':[0.8,1.2],'shear':[0,0,0,0]}
        default_rdparams.update(rdparams)
        self.rdparams = default_rdparams
    def name(self):
        return 'affine'
    def __call__(self, image, label, params=None):
        ''' Affine transformation on image and label
        Args:
        image: np array or PIL, [img_rows, img_cols] 
        label: np array or PIL, [img_rows, img_cols]
        '''
        if params is None:
            params = self._random()
        self.params = params
        if self.params['p']:
            image = Affine(rotate_params=self.params['angle'], translate_params=self.params['translate'], 
                           scale_params=self.params['scale'], shear_params=self.params['shear'], mode='bilinear', 
                           padding_mode='zeros')(image)
            label = Affine(rotate_params=self.params['angle'], translate_params=self.params['translate'], 
                           scale_params=self.params['scale'], shear_params=self.params['shear'], 
                           mode=self.params['label_mode'], padding_mode='zeros')(label)
        return image, label      
    def _random(self):
        params = {}
        params['p'] = random.random() < self.rdparams['p']
        params['label_mode'] = self.rdparams['label_mode']
        params['angle'] = random.randint(self.rdparams['angle'][0],self.rdparams['angle'][1])
        params['translate'] = [random.randint(self.rdparams['translate'][0],self.rdparams['translate'][1]),
                               random.randint(self.rdparams['translate'][2],self.rdparams['translate'][3])]
        params['scale'] = random.random()*(self.rdparams['scale'][1]-self.rdparams['scale'][0]) \
                          + self.rdparams['scale'][0]                               
        params['shear'] = [random.randint(self.rdparams['shear'][0],self.rdparams['shear'][1]),
                           random.randint(self.rdparams['shear'][2],self.rdparams['shear'][3])]
        return params

class _hflip_worker(Worker):
    def __init__(self,rdparams={}):
        super(_hflip_worker,self).__init__(rdparams)
        default_rdparams = {'p':0.5}
        default_rdparams.update(rdparams)
        self.rdparams = default_rdparams
    def name(self):
        return 'hflip'
    def __call__(self, image, label, params=None):
        if params is None:
            params = self._random()
        self.params = params
        if self.params['p']:
            image = image[:,:,::-1]
            label = label[:,:,::-1]
        return image, label   
    def _random(self):
        params = {}
        params['p'] = random.random() < self.rdparams['p']
        return params

class _gamma_worker(Worker):
    def __init__(self,rdparams={}):
        super(_gamma_worker,self).__init__(rdparams)
        default_rdparams = {'p':0.5, 'gamma':[0.7,1.5], 'gain':1}
        default_rdparams.update(rdparams)
        self.rdparams = default_rdparams
    def name(self):
        return 'gamma'
    def __call__(self, image, label, params=None):
        if params is None:
            params = self._random()
        self.params = params
        if self.params['p']:
            image = AdjustContrast(self.params['gamma'])(image)
        return image, label   
    def _random(self):
        params = {}
        params['p'] = random.random() < self.rdparams['p']
        params['gamma'] = random.random()*(self.rdparams['gamma'][1]-self.rdparams['gamma'][0]) \
                          + self.rdparams['gamma'][0]         
        return params

class _noise_worker(Worker):
    def __init__(self,rdparams={}):
        super(_noise_worker,self).__init__(rdparams)
        default_rdparams = {'p':0.5, 'type':'gaussian', 'std':0.1}
        default_rdparams.update(rdparams)
        self.rdparams = default_rdparams
    def name(self):
        return 'noise'
    def __call__(self, image, label, params=None):
        if params is None:
            params = self._random()
        self.params = params
        if self.params['p']:
            if self.params['type'] == 'gaussian':
                image = image + self.params['std']*np.random.randn(image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
            else:
                raise NotImplementedError 
        return image, label   
    def _random(self):
        params = {}
        params['p'] = random.random() < self.rdparams['p']   
        params['std'] = self.rdparams['std']
        params['type'] = self.rdparams['type']
        return params

class _padcrop_worker(Worker):
    def __init__(self,rdparams={}):
        super(_padcrop_worker,self).__init__(rdparams)
        default_rdparams = {'width':400, 'height':400}
        default_rdparams.update(rdparams)
        self.rdparams = default_rdparams
    def name(self):
        return 'padcrop'
    def __call__(self, image, label, params=None):
        if params is None:
            params = self._random()
        self.params = params
        if self.params['width'] > 0 and self.params['height'] > 0:
            image = ResizeWithPadOrCrop([self.params['width'], self.params['height']])(image)
            label = ResizeWithPadOrCrop([self.params['width'], self.params['height']])(label)
        return image, label   
    def _random(self):
        params = {}
        params = self.rdparams
        return params
 

registered_workers = \
{
    'affine':_affine_worker,
    'hflip':_hflip_worker,
    'normalize':_normalize_worker,
    'gamma':_gamma_worker,
    'noise':_noise_worker,
    'padcrop':_padcrop_worker
}
