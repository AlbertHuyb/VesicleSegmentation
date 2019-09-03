import os
import numpy as np
import torch

from config import cfg

#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_mode = cfg.DATASET
print(data_mode)
from datasets.loading_data import loading_data
from datasets.setting import cfg_data 


#------------Prepare Trainer------------
net = cfg.NET
from trainer import Trainer

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data,cfg_data,pwd)
print("begin training.")
cc_trainer.forward()
