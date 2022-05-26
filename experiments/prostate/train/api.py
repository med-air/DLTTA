import copy
import logging
import random
import sys,os

import numpy as np
import pandas as pd
import torch
from .center import Center


class API(object):
    def __init__(self, dataset, device, args, model_trainer):
        """
        dataset: data loaders and data size info
        """
        self.device = device
        self.args = args
        client_num, [source_data, ood_data] = dataset
        print(len(source_data))
        
       

        self.center_list = []
        self.ood_data = ood_data
        
        self.model_trainer = model_trainer
        self._setup_centers(source_data, model_trainer)
        logging.info("############setup ood centers#############")
        self.ood_center = Center(-1,  ood_data, self.args, self.device, model_trainer)
      

    def _setup_centers(self, training_data, model_trainer):
       
        center_idx = 0
        c = Center(center_idx, training_data, self.args, self.device, model_trainer)
        self.center_list.append(c)

    def train(self):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.rounds):

            logging.info("============ round : {}".format(round_idx))

               
            center_idx = 0
            center = self.center_list[center_idx]
            
            w = center.train(copy.deepcopy(w_global))
            w_global = copy.deepcopy(w)
            torch.save(w_global, os.path.join(self.args.save_path, "global_round{}".format(round_idx)))
            self.model_trainer.set_model_params(w_global)
            self._ood_test(round_idx, self.ood_center, self.ood_data, w_global)
            self._test_time_adaptation()
           

    
    def _ood_test(self, round_idx, ood_client, ood_data, w_global):
        logging.info("============ ood_test : {}".format(round_idx))
        metrics = ood_client.ood_test(ood_data, w_global)
        ''' unify key'''
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': '{:.4f}'.format(test_acc), 'test_loss': '{:.4f}'.format(test_loss)}
                   
        logging.info(stats)
        return metrics
        

            
    def _test_time_adaptation(self, w_global=None):  
        metrics = self.ood_center.test_time_adaptation(copy.deepcopy(w_global))
        
        test_acc = metrics["test_acc"]
        test_loss = metrics["test_loss"]
        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        logging.info("############  performance after test time adaptation  #############")    
        logging.info(stats)
        return metrics


    
    

