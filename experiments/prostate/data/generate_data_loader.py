import torch
from torch.utils.data.dataloader import DataLoader

def generate_data_loader(args, client_num, source_set, ood_set):
     
 
     ood_loader = torch.utils.data.DataLoader(ood_set, batch_size=args.batch, shuffle=False)
     source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch, shuffle=True)
     return (client_num, [source_loader, ood_loader])