import argparse
import yaml

prosate = ['BIDMC', 'HK',  'ISBI', 'ISBI_1.5', 'UCL', 'I2CVB', None]
available_datasets = prosate 
   
def set_configs():
     parser = argparse.ArgumentParser()
     parser.add_argument('--log', action='store_true', help='whether to log')
     parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
     parser.add_argument('--early', action='store_true', help='early stop w/o improvement over 10 epochs')
     parser.add_argument('--batch', type = int, default= 1, help ='batch size')
     parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
     parser.add_argument("--target", choices=available_datasets, default=None, help="Target")
     parser.add_argument('--rounds', type = int, default=500, help = 'rounds')
     parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
     parser.add_argument('--save_path', type = str, default='../checkpoint/', help='path to save the checkpoint')
     parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
     parser.add_argument('--gpu', type = str, default="0", help = 'gpu device number')
     parser.add_argument('--seed', type = int, default=0, help = 'random seed')
     parser.add_argument('--client_optimizer', type = str, default='adam', help='local optimizer')
     parser.add_argument('--data', type = str, default='prostate', help='datasets')
     parser.add_argument('--test_time', type = str, default='tent', help='test time adaptation methods')
     parser.add_argument('--debug', action='store_true', help = 'use small data to debug')
     parser.add_argument('--test', action='store_true', help='test on local clients')
     parser.add_argument('--ood_test', action='store_true', help='test on ood client')
     parser.add_argument('--every_save', action='store_true', help='Save ckpt with explicit name every iter')
     
     args = parser.parse_args()
   
     
     return args
     
