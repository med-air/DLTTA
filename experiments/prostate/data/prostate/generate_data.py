from torch.utils.data import dataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch

from .dataset import Prostate
from data.generate_data_loader import generate_data_loader

def load_prostate(args):
    sites = args.source
    ood_site = args.target if args.target is not None else 'HK' 
    client_num = 1
    
    transform = None
   
    ood_set = Prostate(site=ood_site)
    
    source_set = Prostate(site=sites[0])                                
    
    

    dataset = generate_data_loader(args, client_num, source_set, ood_set)

    return dataset
