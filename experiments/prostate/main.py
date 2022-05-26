import sys, os
import logging
import pandas as pd

from torch.utils import data
from torch.utils.data import dataset
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from train.configs import set_configs
from data.prostate.generate_data import load_prostate
from train.api import API
from train.model_trainer_segmentation import ModelTrainerSegmentation



def deterministic(seed):
     cudnn.benchmark = False
     cudnn.deterministic = True
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     

def set_paths(args):
     args.save_path = './checkpoint/'
     if not os.path.exists(args.save_path):
          os.makedirs(args.save_path)

def custom_model_trainer(args):
     
    
     from nets.unet import UNet3D
     model = UNet3D(in_channels=1, out_channels=2, layer_order='cbr')
     model_trainer = ModelTrainerSegmentation(model, args)
     
     return model_trainer

def custom_dataset(args):
     
     datasets = load_prostate(args)
     return datasets

def custom_api(args, model_trainer, datasets):
     device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
     api = API(datasets, device, args, model_trainer)
     return api
     

if __name__ == "__main__":
     args = set_configs()
     args.generalize = False
     deterministic(args.seed)
     set_paths(args)
     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
     log_path = args.save_path.replace('checkpoint', 'log')
     if not os.path.exists(log_path): os.makedirs(log_path)
     log_path = log_path+'/log.txt' if args.log else './log.txt'
     logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
     logging.info(str(args))

     model_trainer = custom_model_trainer(args)
     datasets = custom_dataset(args)
     manager = custom_api(args, model_trainer, datasets)
     if args.ood_test:
            
          global_round = 99     
          ckpt = torch.load(f'/research/dept5/mrjiang/hzyang/3d/checkpoint/prostate/{args.target}/seed0/fedavg_global_round{global_round}')
          
          model_trainer.set_model_params(ckpt)
          print('Finish intialization')
      
          metrics = manager.ood_client.test_time_adaptation(None)
          
     elif args.test:
          from tqdm import tqdm
          rounds = [i for i in range(500)]
          ood_performance = {"before":[]} 
          for epoch in tqdm(rounds):
              ckpt = torch.load(f'/research/pheng4/qdliu/hzyang/prostate/test_time/checkpoint/prostate/I2CVB/seed0/fedavg/fedavg_global_round{epoch}')
              model_trainer.set_model_params(ckpt)
              test_data = datasets[1][-1]
              metrics = model_trainer.test(test_data, manager.device, args) 
              test_acc = metrics["test_acc"]
              ood_performance['before'].append(test_acc)   
              ood_performance_pd = pd.DataFrame.from_dict(ood_performance)  
              ood_performance_pd.to_csv(f"{args.target}_ood_performance_cbr_prostate_center.csv")
               
     else:
          manager.train()
     
     
