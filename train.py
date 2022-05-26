from __future__ import division
import matplotlib
from matplotlib import pyplot as plt
from scipy.io import savemat
import pdb
import argparse
import functools
import warnings
import logging
logging.basicConfig(level=logging.INFO)
import os
import shutil
import time
from scipy import misc
import numpy as np
import json,csv
import time
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
import torch.nn as nn
from datasets import create_dataset
from models import create_model 
from utils.util import setlogger, deterministic
from config import *
from memory import Memory

buffer_size = 20



def configure_model(model):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
    return model


def main():
    warnings.filterwarnings('ignore')
    # set random seed
    deterministic()
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    logger = logging.getLogger('global')
    logger = setlogger(logger,args)
    # build dataset
    # val_loader is a list of dataloader for a list of test subject
    train_loader, val_loader = create_dataset(args)
    logger.info('build dataset done')
    # build model
    model = create_model(args)
    #model.TNet = configure_model(model.TNet)
    #model.ANet = configure_model(model.ANet)
    memory = Memory(buffer_size)
    model.memory = memory
    model.buffer_size = buffer_size
    model.diff = []
    
    logger.info('build model done')
    # logger.info(model)   
    # evaluate
    if args.evaluate:
        logger.info('begin evaluation')
        val_loader[0].dataset.augment = False
        loss = validate(model, val_loader[0], args, 0, logger)
        return loss
    if args.test:
        # put test image in vimg_path 
        logger.info('begin testing')
        model.retri_size = 8
        # train adaptor
        metric_adps = []
        for sub in range(len(val_loader)):
            logger.info('testing subject:{}/{}'.format(sub+1,len(val_loader)))
            val_loader[sub].dataset.augment = args.val_augment
            prev_loss = np.inf
            sub_metric_adp, sub_metric_nadp = [], []
            start_time = time.time() 
            model.set_opt()            
            for epoch in range(args.tepochs):  
                m_loss = 0                                          
                for iters, data in enumerate(val_loader[sub]):               
                    model.set_input(data)
                    loss = model.opt_ANet(epoch)
                    logger.info('[{}/{}][{}/{}] Adaptor Loss: {}'.format(\
                                epoch+1, args.tepochs, iters, len(val_loader[sub]), loss))  
                    m_loss += np.sum(loss)/len(val_loader[sub])
             
            start_time = time.time()   
            # turn off augmentation on test inference
            val_loader[sub].dataset.augment = False 
            # allow 3D metric calculation
            labels, preds = [], []             
            for iters, data in enumerate(val_loader[sub]):
                model.set_input(data)
                _metric, pred = model.test(return_pred=True)               
                labels.append(model.label)
                preds.append(pred)
                
                
            labels = torch.stack(labels)
            preds = torch.stack(preds)

            _metric_adps = model.cal_metric3d(preds.view(-1,preds.shape[-3],preds.shape[-2], preds.shape[-1]),\
                                               labels.view(-1,labels.shape[-2], labels.shape[-1]))

            metric_adps.append(_metric_adps)
            
            
        metric_adps = np.vstack(metric_adps)
        logger.info('Overall 3D mean metric adp:\n{}[{}]'.\
                    format(str(np.nanmean(metric_adps,axis=0)).replace('\n',''),\
                           np.nanmean(metric_adps)))
             
        return   
    
if __name__ == '__main__':
    main()

