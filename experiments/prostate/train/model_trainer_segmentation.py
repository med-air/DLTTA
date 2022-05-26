import logging

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from .model_trainer import ModelTrainer
from utils.loss import DiceLoss, entropy_loss
import copy
import numpy as np
import random
from test_dataset import Prostate
from torch.utils.data.dataloader import DataLoader
def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)


class ModelTrainerSegmentation(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    @torch.enable_grad()    
    def test_time(self, test_data, device, args):
        test_set = Prostate(args.target)
        test_data = DataLoader(test_set, batch_size=1, shuffle=True)
        deterministic(args.seed)
        metrics = {
            'test_acc': 0,
            'test_loss': 0,
        }
        best_dice = 0.
        dice_buffer = []
        model = self.model
        model_adapt = copy.deepcopy(model)
        model_adapt.to(device)
        model_adapt.train()
        for m in model_adapt.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        var_list = model_adapt.named_parameters()
        update_var_list = []
        update_name_list = []
        names = []

        for idx, (name, param) in  enumerate(var_list):
            param.requires_grad_(False)
            names.append(name)
            if "batchnorm" in name:
                param.requires_grad_(True)
                update_var_list.append(param)
                update_name_list.append(name)
        params = model_adapt.parameters()
        optimizer = torch.optim.Adam(update_var_list, lr=1e-3, betas=(0.9, 0.999))
        criterion = DiceLoss().to(device)
        loss_all = 0
        test_acc = 0.
        
        for epoch in range(1):
            loss_all = 0
            test_acc = 0.
            
            deterministic(args.seed)
            for step, (data, target) in enumerate(test_data):
                deterministic(args.seed)
                data = data.to(device)
                target = target.to(device)
                output = model_adapt(data)              
                loss_entropy_before = entropy_loss(output, c=2)        
                all_loss = loss_entropy_before
                weight = 1
                all_loss = weight*all_loss
                #print(all_loss)
                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()
                output = model_adapt(data)       
                loss = criterion(output, target)
                loss_all += loss.item()   
        loss = loss_all / len(test_data)
        acc = 1 - loss  
        metrics['test_loss'] = loss
        metrics["test_acc"] = acc
        return metrics
        
    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = DiceLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, amsgrad=True)

        epoch_loss = []
        epoch_acc = []
        for epoch in range(args.wk_iters):
            batch_loss = []
            batch_acc = []
            for batch_idx, (x, labels) in enumerate(train_data):
                model.zero_grad()
                x, labels = x.to(device), labels.to(device)
                
                #print(x.shape)
                log_probs = model(x)
                loss = criterion(log_probs, labels)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_acc.append(1-loss.item())
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(sum(batch_acc) / len(batch_acc))
            logging.info('Client Index = {}\tEpoch: {}\tAcc:{:.4f}\tLoss: {:.4f}'.format(
                self.id, epoch, sum(epoch_acc) / len(epoch_acc),sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args, ood=False):
        model = copy.deepcopy(self.model)

        model.to(device)
        if ood:
            model.train()
        else:
            model.eval()

        metrics = {
            'test_acc': 0,
            'test_loss': 0,
            'test_hd': 0,
        }
        test_set = Prostate(args.target)
        test_data = DataLoader(test_set, batch_size=1, shuffle=True)
        criterion = DiceLoss().to(device)
        test_epoches = 1
        with torch.no_grad():
            for test_epoch in range(test_epoches):
                for batch_idx, (x, target) in enumerate(test_data):
                    x = x.to(device)
                    target = target.to(device)
                    pred = model(x)
                    loss = criterion(pred, target)
    
                    acc = 1 - loss.item()
                    #print(acc)
    
                    metrics['test_loss'] += loss.item() 
                    metrics['test_acc'] += acc
                
        metrics["test_loss"] = metrics["test_loss"] / (len(test_data)*test_epoches)
        metrics["test_acc"] = metrics["test_acc"] / (len(test_data)*test_epoches)
        
        return metrics
    

