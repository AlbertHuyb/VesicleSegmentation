import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import sys
import random
import torch
import cv2
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import utils.transforms as own_transforms
import pandas as pd
import pdb
from utils.CC import CrowdCounter
from utils.config import cfg
from utils.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
from skimage.measure import label
import numpy as np
from config import *

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = os.path.join(OUTPUT_DIR,'result/',sys.argv[2])

if not os.path.exists(exp_name):
    os.mkdir(exp_name)

if not os.path.exists(exp_name+'/pred'):
    os.mkdir(exp_name+'/pred')

if not os.path.exists(exp_name+'/gt'):
    os.mkdir(exp_name+'/gt')

if not os.path.exists(exp_name+'/mask'):
    os.mkdir(exp_name+'/mask')

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = os.path.join(OUTPUT_DIR,'full',sys.argv[2])

VESICLE_PATH = os.path.join(MODEL_DIR,sys.argv[1])
model_path = VESICLE_PATH

def main():
#     file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]      
    file_list = [filename for root,dirs,filename in os.walk(dataRoot)]                                     

    test(file_list[0], model_path)
   

def test(file_list, model_path):

    net = CrowdCounter(cfg.GPU_ID,cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()


    f1 = plt.figure(1)

    gts = []
    preds = []

    for filename in file_list:
        print filename
        # imgname = dataRoot + '/img/' + filename
        imgname = os.path.join(dataRoot,filename)
        filename_no_ext = filename.split('.')[0]

        # denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        # den = pd.read_csv(denname, sep=',',header=None).values
        # den = den.astype(np.float32, copy=False)

        img = Image.open(imgname)
        shape_orig = np.array(img).shape
        
        cv_image = np.array(img)
        input_img = cv2.resize(cv_image,(1024,768))
        img = Image.fromarray(input_img)
        print(np.array(img).shape)        
        
        if img.mode == 'L':
            img = img.convert('RGB')


        img = img_transform(img)

        # gt = np.sum(den)
        with torch.no_grad():
            img = Variable(img[None,:,:,:]).cuda()
            pred_map = net.test_forward(img)

        # np.save(exp_name+'/pred/'+filename_no_ext+'.npy',pred_map.squeeze().cpu().numpy()/100.)
        # sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
        # sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':den})

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
        
        pred_map = cv2.resize(pred_map,(shape_orig[1],shape_orig[0]))
        print(pred_map.shape)
        np.save(exp_name+'/pred/'+filename_no_ext+'.npy',pred_map/100.)
        pred_mask = pred_map/100.0 >= 0.3
        pred_label = label(pred_mask)
        count = np.max(pred_label)
        # sio.savemat(exp_name+'/mask/'+filename_no_ext+'.mat',{'data':pred_mask})
        cv2.imwrite(exp_name+'/mask/'+filename_no_ext+'-count-'+str(count)+'.png',(pred_mask*255).astype('float'))
        

        pred = np.sum(pred_map)/100.0
        pred_map = pred_map/np.max(pred_map+1e-20)
        
        # den = den/np.max(den+1e-20)

        
        # den_frame = plt.gca()
        # plt.imshow(den, 'jet')
        # den_frame.axes.get_yaxis().set_visible(False)
        # den_frame.axes.get_xaxis().set_visible(False)
        # den_frame.spines['top'].set_visible(False) 
        # den_frame.spines['bottom'].set_visible(False) 
        # den_frame.spines['left'].set_visible(False) 
        # den_frame.spines['right'].set_visible(False) 
        # plt.savefig(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
        #     bbox_inches='tight',pad_inches=0,dpi=150)

        # plt.close()
        
        # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})

        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False) 
        pred_frame.spines['bottom'].set_visible(False) 
        pred_frame.spines['left'].set_visible(False) 
        pred_frame.spines['right'].set_visible(False) 
        plt.savefig(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=150)

        plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})
        # pred_map = cv2.resize(pred_map,(shape_orig[1],shape_orig[0]))
        # diff = den-pred_map
 
        # diff_frame = plt.gca()
        # plt.imshow(diff, 'jet')
        # plt.colorbar()
        # diff_frame.axes.get_yaxis().set_visible(False)
        # diff_frame.axes.get_xaxis().set_visible(False)
        # diff_frame.spines['top'].set_visible(False) 
        # diff_frame.spines['bottom'].set_visible(False) 
        # diff_frame.spines['left'].set_visible(False) 
        # diff_frame.spines['right'].set_visible(False) 
        # plt.savefig(exp_name+'/'+filename_no_ext+'_diff.png',\
        #     bbox_inches='tight',pad_inches=0,dpi=150)

        # plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})
                     



if __name__ == '__main__':
    main()




