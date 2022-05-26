from models.basemodel import AdaptorNet
from models.segmodel import SegANet

import pdb
import logging
logger = logging.getLogger('global')
def create_model(args):
    """Create a model and load its weights if given
    """
   
    adaptorNet = SegANet
       
    model = adaptorNet(args)
    if args.resume_T:
        model.load_nets(args.resume_T, name='tnet')
    if args.resume_AE:
        model.load_nets(args.resume_AE, name='aenet')           
    return model