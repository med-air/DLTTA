import argparse
parser = argparse.ArgumentParser(description='PyTorch parser')
# training 
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--sepochs', default=0, type=int, metavar='N',
                    help='number of total epochs to initialize adaptor')  
parser.add_argument('--tepochs', default=1, type=int, metavar='N',
                    help='number of total epochs to during test')                    
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--segloss', dest='segloss', default='ce',type=str,
                    help='wce,  ce, dice. segmentation loss')
parser.add_argument('--segaeloss', dest='segaeloss', default='mse',type=str,
                    help='segmentation output ae loss')         
parser.add_argument('--usegt', dest='use_gt', action='store_true', default=False,
                    help='use gt for last segmentation map autoencoder training')  
parser.add_argument('--usedtta', dest='usedtta', action='store_true', default=False,
                    help='use dtta in MEDIA paper')                           
parser.add_argument('--segsoftmax', dest='segsoftmax', action='store_true', default=False,
                    help='use softmax before last autoencoder for segmentation')            
parser.add_argument('--tlr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for TNet')
parser.add_argument('--aelr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for AENet')
parser.add_argument('--alr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for ANet')
parser.add_argument('--or_weight', default=0.05, type=float,
                    metavar='LR', help='or_weight')                                                                                           
parser.add_argument('--wt',dest='weights', type=lambda x: list(map(float, x.split(','))),
                    help='weights in training ae net')           
parser.add_argument('--wo', default=10.0, type=float,
                    help='orthogonal weights in training ANet')                                  
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='test model on validation set')   
parser.add_argument('--resume_T', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_AE', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)') 
# network  
parser.add_argument('--trainer',dest='trainer', default='tnet', 
                    type=str, help='select which net to train') 
parser.add_argument('--task',dest='task', default='syn', 
                    type=str, help='select task (syn:synthesis or seg:segmentation)') 
parser.add_argument('--seq',dest='seq', type=lambda x: list(map(int, x.split(','))),
                    help='the 1x1 conv seq to be used in A-Net')
parser.add_argument('--td',dest='tnet_dim', type=lambda x: list(map(int, x.split(','))),
                    help='task net input, encoder, output channels')   
parser.add_argument('--ad', dest='aenet_dim', default=64, type=int,
                    help = 'starting feature channel in auto-encoder')                 
parser.add_argument('--na', dest='no_adpt', action='store_true',
                    help = 'not using pre-adaptation')
parser.add_argument('--config', dest='config', default='config.json',
                    help='hyperparameter in json format')
# datasets                
parser.add_argument('--label_path', dest='label_path', default='../data/label/',type=str,
                    help='path to the label')
parser.add_argument('--label_ext', dest='label_ext', default='json',type=str,
                    help='label extension')
parser.add_argument('--img_path', dest='img_path', default='../data/image/',type=str,
                    help='path to the image')
parser.add_argument('--img_ext', dest='img_ext', default='png',type=str,
                    help='image extension')
parser.add_argument('--vlabel_path', dest='vlabel_path', default='',type=str,
                    help='path to the validation label')
parser.add_argument('--vimg_path', dest='vimg_path', default='',type=str,
                    help='path to the validation image')
parser.add_argument('--sub_name', dest='sub_name', default='',type=str,
                    help='path to the txt file name containing subject unique ID')                    
parser.add_argument('--split',dest='split', type=lambda x: list(map(int, x.split(','))),
                    help='the start and end index for validation dataset')
# preprocessing         
parser.add_argument('--ps',dest='pad_size', type=lambda x: list(map(int, x.split(','))),
                    help='padding all the input image to this size')     
parser.add_argument('--scs',dest='scale_size', type=lambda x: list(map(int, x.split(','))),
                    help='interpolate all the input image to this size')   
parser.add_argument('--an', dest='add_noise', action='store_true',
                    help = 'add gaussian noise in preprocessing')     
# augmentation  
parser.add_argument('--fnoise', dest='feat_noise', action='store_true', default=False,
                    help='use feature noise to train denoising auto-encoders') 
parser.add_argument('--aprob', dest='aug_prob', type=float, default=0,
                    help='use augmentation during tnet and aenet training')                                   
parser.add_argument('--aangle',dest='affine_angle', type=lambda x: list(map(float, x.split(','))),
                    default='-30,30', help='affine transformation angle range')    
parser.add_argument('--atrans',dest='affine_translate', type=lambda x: list(map(float, x.split(','))),
                    default='-10,10,-10,10', help='affine transformation translation range')    
parser.add_argument('--ascale',dest='affine_scale', type=lambda x: list(map(float, x.split(','))),
                    default='0.8,1.2',help='affine transformation scaling range')     
parser.add_argument('--agamma',dest='gamma', type=lambda x: list(map(float, x.split(','))),
                    default='0.7,1.5', help='gamma scaling range')       
parser.add_argument('--anoise',dest='noise_std', type=float,default=0.1,
                    help='gaussian noise std') 
parser.add_argument('--width',dest='width', type=int,default=400,
                    help='centor crop width') 
parser.add_argument('--height',dest='height', type=int,default=400,
                    help='centor crop height') 
parser.add_argument('-vaug', dest='val_augment', action='store_true', default=False,
                    help='use augmentation during test time training')               
# logging                                                                               
parser.add_argument('--results_dir', dest='results_dir', default='results_dir',
                    help='results dir of output')
parser.add_argument('--ss', dest='save_step', default=5, type=int,
                    help = 'The step of epochs to save checkpoints and validate')
parser.add_argument('--saveimage','--si', dest='saveimage', action='store_true',
                    help='save image with surfaces and layers')
parser.add_argument('--dpi', dest='dpi', type=int, default=100, help='dpi of saved image')
