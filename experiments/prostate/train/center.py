import logging
import torch
import os
import copy

class Center:
    def __init__(self, client_idx, training_data, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = training_data
      
       
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
       

   

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        
        return weights

    
    def ood_test(self, ood_data, w_global):
        self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test(ood_data, self.device, self.args, True)
        return metrics
    
                
    def test_time_adaptation(self, w_global):
        if w_global != None:
            self.model_trainer.set_model_params(w_global)
        metrics = self.model_trainer.test_time(self.local_training_data, self.device, self.args)
        return metrics
        