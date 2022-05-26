import logging

import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms



import mbtt
from camelyon17_dataset import CamelyonDataset

from conf import cfg, load_cfg_fom_args
from models import  initialize_model
import random 
import numpy as np

from tqdm import tqdm

import pandas as pd

logger = logging.getLogger(__name__)

err_rates = []

torch.set_printoptions(precision=5)


def evaluate(description):
    load_cfg_fom_args(description)
    # configure model
    base_model = initialize_model('densenet121', 2)
    checkpoint = torch.load('best.pth')
    base_model.load_state_dict(checkpoint['state_dict'])
    base_model = torch.nn.DataParallel(base_model)
    base_model = base_model.cuda()
   
    model = setup_mbtt(base_model)
   
    try:
        model.reset()
        logger.info("resetting model")
    except:
        logger.warning("not resetting model")
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform=transforms.Compose([      transforms.Resize((96, 96)),
                                        transforms.ToTensor(),
                                        normalize,
                                          ])
    dataset = CamelyonDataset(None, transform, 'test')
    print(len(dataset))
    test_loader = data.DataLoader(dataset,
                                  batch_size=200,
                                  shuffle=True,
                                  num_workers=8)
    print('done')
    acc = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            model.gt = y
            x, y = x.cuda(), y.cuda()
            output = model(x)
            acc += (output.max(1)[1] == y).float().sum()
    print(acc)
    
    
    acc = acc.item() / len(dataset)
    err = acc
    err_rates.append(err)
    avg_err = sum(err_rates)/len(err_rates)
    logging.info("\n error: {:6f}".format(avg_err))


def setup_mbtt(model):
    model = mbtt.configure_model(model)
    
    params, param_names = mbtt.collect_params(model)
    optimizer = setup_optimizer(params)
    mbtt_model = mbtt.Tent(model, optimizer, cfg, 
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return mbtt_model


def setup_optimizer(params):
    
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError

def setup_determinism(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    setup_determinism(0)
    evaluate('evaluate')
