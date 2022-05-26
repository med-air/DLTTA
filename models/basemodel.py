import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import copy
import pdb
from models.backends import UNet, ConvBlock
from models.adapmodel import ANet, DTTAnorm
from utils.util import ncc, l2_reg_ortho
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import logging
import tifffile as tiff
from functools import partial
logger = logging.getLogger('global')
def tonp(x):
    return x.squeeze().data.cpu().numpy()
    
    
def magic_combine(x, dim_begin, dim_end):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)


class AdaptorNet(nn.Module):
    """Base model with UNet backbone
    """
    def __init__(self, opt):
        super(AdaptorNet,self).__init__()
        self.opt = opt
        self.def_TNet()
        self.def_AENet()
        self.def_ANet()
        self.def_loss()
        self.set_opt()
    def set_opt(self):
        """Move models to GPU and set optimizers.
        """
        # set TNet optimizers
        self.TNet.cuda()
        self.optimizer_TNet = torch.optim.Adam(self.TNet.parameters(), 
                                               lr=self.opt.tlr)
        # set AENet optimizers
        params = [] 
        for subnets in self.AENet:
            subnets.cuda()
            params.extend(list(subnets.parameters()))
        self.optimizer_AENet = torch.optim.Adam(params, self.opt.aelr)
        # set ANet optimizers
        self.ANet.cuda()
        self.optimizer_ANet = torch.optim.SGD(self.ANet.parameters(),
                                               self.opt.alr)
    def def_TNet(self):
        """Define Task Net for synthesis, segmentation e.t.c.
        """
        dims = self.opt.tnet_dim
        self.TNet = UNet(dims[0],dims[1:-1],dims[-1],skip=True,isn=False)   
    def def_AENet(self):
        """Define Auto-Encoder for training on source images
           Network definition could be moved to config.json
        6 level AE list: Dependent on TNet structure
        level 0: input image(seq: None),
                 l0-down-feature(seq:0) + l0-up-feature(seq:7),
                 output image(seq: None)
        level 1: l1-down-feature(seq:1) + l1-up-feature(seq:6)
        level 2: l2-down-feature(seq:2) + l2-up-feature(seq:5)
        level 3: l3-down-feature(seq:3) + l3-up-feature(seq:4)
        """       
        # Examples: the matching index of 4-level UNet features 
        # self.tnet_dim = [1,64,64,64,64,1]         
        # self.AENetMatch = [[0],[1,-2],[2,-3],[3,-4],[4,-5],[-1]] 
        self.AENetMatch = [[0]]
        for i in range(1,len(self.opt.tnet_dim)-1):
            self.AENetMatch += [[i,-i-1]]
        self.AENetMatch += [[-1]]   
        self.AENet = []
        n0 = self.opt.aenet_dim
        for _ in self.AENetMatch:
            if len(_) == 1:
                dims = self.opt.tnet_dim[_[0]]
                self.AENet += [UNet(inplane=dims, midplane=[n0//2,n0//4,n0//8], \
                                outplane=dims, skip=False, isn=True)]  
            else:
                dims = self.opt.tnet_dim[_[0]] + self.opt.tnet_dim[_[1]]
                self.AENet += [UNet(inplane=dims, midplane=[n0,n0//2,n0//4], \
                                    outplane=dims, skip=False, isn=True)]  
    def def_ANet(self):
        """Define Adaptor Net for domain adapt
        """
        adp = None
        if self.opt.usedtta:
            adp = DTTAnorm()
        self.ANet = ANet(adpNet=adp, tnet_dim=self.opt.tnet_dim, 
                         seq=self.opt.seq)  
    def def_loss(self):
        """Define training loss
        examples:
            self.TLoss = nn.MSELoss()
            self.AELoss = nn.MSELoss()
            self.AELoss_out = nn.MSELoss()
        """
        pass
    def entropy(self,x):
        """Entropy loss
        pred: [batch, channel, H, W], before softmax
        """
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(1).mean()
        return b
    def compute_kl(self, mu):
        ''''
        pseudo KL loss
        taken from: https://github.com/NVlabs/MUNIT/blob/master/trainer.py
        '''
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss    
    def set_requires_grad(self, nets, requires_grad=False, cuda=True):
        """Set requies_grad=Fasle and move to cuda
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                if cuda:
                    net.cuda()
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def set_data_pre(self,**kwargs):
        """Preprocess the input image and labels
           Supports cropping/padding/scaling
           Images from different dataset needs to have the same
           physical resolution and pre-registered to some common
           space.
        """    
        if 'image' not in self.__dict__: return 
        if self.opt.pad_size:
            h_n, w_n = self.opt.pad_size
            h,w = self.image.shape[-2:]
            # crop in the middel
            if h > h_n:
                self.image = self.image[:,:,(h-h_n)//2:(h-h_n)//2+h_n, :]
                self.label = self.label[:,(h-h_n)//2:(h-h_n)//2+h_n, :]
            if w > w_n:
                self.image = self.image[:,:,:,(w-w_n)//2:(w-w_n)//2+w_n]
                self.label = self.label[:,:,(w-w_n)//2:(w-w_n)//2+w_n]
            h,w = self.image.shape[-2:]
            # padding 
            pd = tuple(map(int,[(w_n-w)//2, w_n-w-(w_n-w)//2, (h_n-h)//2, h_n-h-(h_n-h)//2]))
            self.image = nn.ReplicationPad2d(pd)(self.image)
            types = self.label.dtype
            self.label = nn.ReplicationPad2d(pd)(self.label.unsqueeze(1).to(torch.float)).squeeze(1).to(types)
        if self.opt.scale_size:
            upl = nn.Upsample(tuple(map(int,self.opt.scale_size)), mode='bilinear')
            self.image = upl(self.image)
            self.label = upl(self.label.unsqueeze(1)).squeeze(1)
        if self.opt.add_noise:
            self.image += 0.3*torch.randn(self.image.shape).cuda()
    def set_input(self, data):
        """Unpack input data and perform necessary pre-processing steps.
        self.image: [batch, 1, img_rows, img_cols]
        self.label: [batch, img_rows, img_cols]
        self.filename: str
        """    
        self.image = torch.stack(tuple(data['data'])).cuda()
        self.label = torch.stack(tuple(data['label'])).cuda()
        self.filename = data['filename']    
        self.set_data_pre()
    def opt_TNet(self):      
        """Optimize Task Net seperately
        """  
        self.set_requires_grad(self.TNet,True)
        self.TNet.train()
        pred = self.TNet.forward(self.image,side_out=False)
        self.optimizer_TNet.zero_grad()
        loss = self.TLoss(pred, self.label.long())
        loss.backward()
        self.optimizer_TNet.step()
        return loss.data.item()
    def opt_AENet(self):
        """Optimize Auto-Encoder seperately
        """
        self.set_requires_grad(self.TNet,False)
        self.set_requires_grad(self.AENet,True)
        self.TNet.eval()
        for subnets in self.AENet:
            subnets.train()
        side_out = self.TNet(self.image,side_out=True)
        self.optimizer_AENet.zero_grad()
        loss = torch.tensor(0.0).cuda()
        loss_list = []
        weights = self.opt.__dict__.get('weights', [1]*len(self.AENet))                
        for _ in range(len(self.AENet)):
            if weights[_] == 0:
                level_loss = torch.tensor(0.0)
            else:
                if len(self.AENetMatch[_]) == 2:
                    # concatenate features from the same level
                    side_out_cat_orig = torch.cat([side_out[self.AENetMatch[_][0]], \
                                              side_out[self.AENetMatch[_][1]]], dim=1)
                    side_out_cat = torch.cat([self.addnoise(side_out[self.AENetMatch[_][0]]), \
                                            self.addnoise(side_out[self.AENetMatch[_][1]])], dim=1)
                else:
                    # use seperate features
                    side_out_cat_orig = side_out[self.AENetMatch[_][0]]
                    side_out_cat = self.addnoise(side_out[self.AENetMatch[_][0]])
                if ('seg' in self.opt.task) and _ == len(self.AENetMatch) - 1:
                    if self.opt.segsoftmax:
                        # if use softmax, the label auto-encoder input is the segmentation maps
                        if self.opt.use_gt:
                            # if use gt, the AE training is not related to task model
                            label_ = side_out_cat_orig.clone().detach().zero_()
                            label_.scatter_(1, self.label.long().unsqueeze(1), 1)
                            label_noise = self.addnoise(label_)
                            ae_out = self.AENet[_](label_noise,side_out=False)                            
                            level_loss = weights[_]*self.AELoss_out(F.softmax(ae_out,1), label_)
                        else:
                            ae_out = self.AENet[_](F.softmax(side_out_cat,1),side_out=False)
                            level_loss = weights[_]*self.AELoss_out(F.softmax(ae_out,1), F.softmax(side_out_cat_orig,1))
                    else:
                        ae_out = self.AENet[_](side_out_cat,side_out=False)
                        level_loss = weights[_]*self.AELoss_out(ae_out, side_out_cat_orig)
                else:                                             
                    if _ == len(self.AENetMatch) - 1 and self.opt.use_gt:
                        label_noise = self.addnoise(self.label.unsqueeze(1))
                        ae_out = self.AENet[_](label_noise,side_out=False)  
                        level_loss = weights[_]*self.AELoss(ae_out, self.label.unsqueeze(1))
                    else: 
                        ae_out = self.AENet[_](side_out_cat,side_out=False)
                        level_loss = weights[_]*self.AELoss(ae_out, side_out_cat_orig)
                # ae_out = self.AENet[_](side_out_cat,side_out=False)
                # level_loss = weights[_]*self.AELoss(ae_out, side_out_cat_orig)
            loss += level_loss
            loss_list.append(level_loss.data.item())
        loss.backward()
        self.optimizer_AENet.step()
        return loss_list
    def opt_ANet(self, epoch, stable=False):
        """Optimize Adaptor
        """
        with torch.no_grad():
            self.TNet.train()
            features = self.TNet(self.image, adaptive=True)
            pred_na =  self.TNet(self.image)
       
        memory_size = self.memory.get_size()
        if memory_size > self.retri_size:
            with torch.no_grad():                
                retrieved_batches = self.memory.get_neighbours(features.cpu().numpy(), k=8) 
                pseudo_past_logits = retrieved_batches.cuda()
                pseudo_current_logits = pred_na
                pseudo_past_labels = nn.functional.softmax(pseudo_past_logits, dim=1)
                pseudo_current_labels = nn.functional.softmax(pseudo_current_logits/4, dim=1)
                pseudo_past_labels = pseudo_past_labels.squeeze()
                pseudo_current_labels = pseudo_current_labels.squeeze()
                pseudo_past_labels = magic_combine(pseudo_past_labels, 1, 3)
                pseudo_current_labels = magic_combine(pseudo_current_labels, 1, 3)
                pseudo_past_labels = pseudo_past_labels.T
                pseudo_current_labels = pseudo_current_labels.T
                diff_loss = F.kl_div((pseudo_past_labels).log(), pseudo_current_labels, None, None, 'none') + F.kl_div((pseudo_current_labels).log(), pseudo_past_labels, None, None, 'none')
                diff_loss = torch.sum(diff_loss, dim=1)
                diff_loss = diff_loss.cpu().numpy().tolist()
                diff_loss = sum(diff_loss) / len(diff_loss)
                diff_loss = np.exp(diff_loss)
                self.diff.append(diff_loss)
                for param_group in self.optimizer_ANet.param_groups:
                    param_group['lr'] = diff_loss * 1e-3
        else:
            for param_group in self.optimizer_ANet.param_groups:
                param_group['lr'] = 1e-3 * memory_size
        self.set_requires_grad([self.TNet] + self.AENet, False) 
        self.set_requires_grad(self.ANet, True)        
        self.TNet.eval()
        for subnets in self.AENet:
            subnets.eval()        
        self.ANet.train() 
        loss = torch.tensor(0.0).cuda()
        loss_list = []
        weights = self.opt.__dict__.get('weights', [1]*len(self.AENet))    
        orthw = self.opt.or_weight
        
        # For feature level adaptation
        side_out = self.ANet(self.image, self.TNet, self.AENet[0], side_out=True)  
        if stable:
            loss += self.AELoss(side_out[0],self.image)
            loss_list.append(loss.data.item())      
        else: 
            for _ in range(len(self.AENet)):
                """Keep this part the same with opt_AENet"""
                if weights[_] == 0:
                    level_loss = torch.tensor(0).cuda()
                else:
                    if len(self.AENetMatch[_]) == 2:
                        # concatenate features from the same level
                        side_out_cat = torch.cat([side_out[self.AENetMatch[_][0]], \
                                                  side_out[self.AENetMatch[_][1]]], dim=1)
                    else:
                        # use seperate features
                        side_out_cat = side_out[self.AENetMatch[_][0]]
                    if ('seg' in self.opt.task) and _ == len(self.AENetMatch) - 1:
                        if self.opt.segsoftmax:
                            ae_out = self.AENet[_](F.softmax(side_out_cat,1),side_out=False)
                            level_loss = weights[_]*self.AELoss_out(F.softmax(ae_out,1), F.softmax(side_out_cat,1))
                        else:
                            ae_out = self.AENet[_](side_out_cat,side_out=False)
                            level_loss = weights[_]*self.AELoss_out(ae_out, side_out_cat)                            
                    else:                                             
                        ae_out = self.AENet[_](side_out_cat,side_out=False)
                        level_loss = weights[_]*self.AELoss(ae_out, side_out_cat)
                    # ae_out = self.AENet[_](side_out_cat,side_out=False)
                    # level_loss = weights[_]*self.AELoss(ae_out, side_out_cat)
                
                loss += orthw* level_loss
                loss_list.append(level_loss.data.item())    
        
        org_loss = orthw*l2_reg_ortho(self.ANet.conv)   
        loss += org_loss
        loss_list.append(org_loss.data.item())
        self.optimizer_ANet.zero_grad()
        loss.backward()
        self.optimizer_ANet.step()
        
        
        with torch.no_grad():
            self.TNet.train()
            feature = self.TNet(self.image, adaptive=True)
            pred_na =  self.TNet(self.image)
            self.memory.push(feature.cpu().numpy(), pred_na.cpu().numpy())   
        
        
        return loss_list
    def plot(self,image,ids):
        """Plot and save image
        Args:
            image: [channel, img_rows, img_cols] for segmentation masks
                   [img_rows, img_cols] for synthesis
        """
        os.makedirs(os.path.join(self.opt.results_dir,'image'), exist_ok=True)
        ext = ids.split('.')[-1]
        if ext == 'tif':        
            if len(image.shape) > 2:
                image = np.argmax(image,axis=0)
            tiff.imsave(os.path.join(self.opt.results_dir,'image',ids), image)
        else:
            fig = plt.figure(1)
            height = float(image.shape[-2])
            width = float(image.shape[-1])        
            fig.set_size_inches(width/height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if len(image.shape) > 2:
                ax.imshow(np.argmax(image,axis=0))
            else:
                ax.imshow(image,cmap='gray')
            fig.savefig(os.path.join(self.opt.results_dir,'image',ids),dpi=height)
            plt.close(fig)
    def plot_hist(self,image,ids, xlim=[-1.5,3]):
        """Plot joint histogram
        Args:
            image: list of images
        """
        os.makedirs(os.path.join(self.opt.results_dir,'image'), exist_ok=True)
        fig, ax = plt.subplots(1, 1)
        if len(image) > 1:
            ax.hist2d(image[0].flatten(),image[1].flatten(), bins=50, normed=True,
                      cmax=1)
            ax.set_xlim(xlim)
            ax.set_ylim(xlim)
        else:
            ax.hist(image[0].flatten(), bins=50, density=True, facecolor='g')
            ax.set_xlim(xlim)
            ax.set_ylim([0,1])
        fig.tight_layout()
        fig.savefig(os.path.join(self.opt.results_dir,'image',ids),dpi=self.opt.dpi)
    def test(self):
        """Test using ANet and TNet
        """
        pass      
    def addnoise(self, feat):
        """ Add noise to features for auto-encoders
        feats [batch, channel, H, W]
        """
        # read config from self.opt
        # random permute features
        if self.opt.feat_noise == False:
            return feat
        blks = [16, int(np.ceil(feat.shape[3]/feat.shape[2])) * 16]
        ratio = 0.25
        radius = [feat.shape[2]//blks[0] + 1, feat.shape[3]//blks[1] + 1]  
        nums = np.round(blks[0] * blks[1] * ratio * ratio)
        wrong_labels = copy.deepcopy(feat)
        for i in range(feat.shape[0]):
            for _ in range(np.random.randint(nums)):
                rx = np.random.randint(1, radius[0] + 1)
                ry = np.random.randint(1, radius[1] + 1)
                mcx = np.random.randint(rx+1, feat.shape[2]-rx-1)
                mcy = np.random.randint(ry+1, feat.shape[3]-ry-1)
                mcx_src = np.random.randint(rx+1, feat.shape[2]-rx-1)
                mcy_src = np.random.randint(ry+1, feat.shape[3]-ry-1)
                wrong_labels[i, :, mcx-rx:mcx+rx, mcy-ry:mcy+ry] = feat[i, :, mcx_src-rx:mcx_src+rx, mcy_src-ry:mcy_src+ry]
        return wrong_labels

    def eval(self):
        """Evaluation on TNet and AENet
        """
        self.TNet.eval()
        with torch.no_grad():
            pred = self.TNet.forward(self.image)
            if 'syn' in self.opt.task:
                loss = self.TLoss(pred, self.label.unsqueeze(1))
            else:
                loss = self.TLoss(pred, self.label.long())
        if self.opt.saveimage:
            batch_size = self.image.shape[0]       
            for b_ix in range(batch_size):
                ids = os.path.split(self.filename[b_ix])[-1].split('.')[0]
                self.plot(tonp(pred[b_ix]), ids + '_pred.png')
                self.plot(tonp(self.image[b_ix]), ids + '_image.png')
                self.plot(tonp(self.label[b_ix]), ids + '_label.png')
        return loss
    def save_nets(self, epoch):
        """Save network weights
        """
        os.makedirs(os.path.join(self.opt.results_dir, 'checkpoints'), exist_ok=True)
        save_path = os.path.join(self.opt.results_dir, 'checkpoints', 
                                 self.opt.trainer + '_checkpoint_e%d.pth' % (epoch))
        save_dicts = {'epoch': epoch}
        if self.opt.trainer == 'tnet':
            save_dicts['state_dict'] = self.TNet.cpu().state_dict()
            save_dicts['optimizer'] = self.optimizer_TNet.state_dict()
        else:       
            save_dicts['state_dict'] = [_.cpu().state_dict() for _ in self.AENet]
            save_dicts['optimizer'] = self.optimizer_AENet.state_dict()
        torch.save(save_dicts, save_path)             
    def load_nets(self, checkpoint, name='tnet'):
        """Load network weights
        """
        ckpt = torch.load(checkpoint)
        state_dict = ckpt['state_dict']
        optimizer = ckpt['optimizer']
        if name is 'tnet':        
            self.TNet.load_state_dict(state_dict)  
            self.optimizer_TNet.load_state_dict(optimizer)
            logger.info('TNet weights loaded')
        elif name is 'aenet':
            for _ in range(len(self.AENet)):
                self.AENet[_].load_state_dict(state_dict[_])
            self.optimizer_AENet.load_state_dict(optimizer)
            logger.info('AENet weights loaded')         
        else:
            raise TypeError('Unknown model name')

