import os
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
from matplotlib import pyplot as plt
from config import *

WSHED_DIR = os.path.join(OUTPUT_DIR,'wshed_result')

OUT_DIR = os.path.join(OUTPUT_DIR,'vast_volume')

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

for num in range(19):
# for num in range(12,13):
    # num=12
    indir = os.path.join(WSHED_DIR,str(num),'data')
    outdir = os.path.join(OUT_DIR,str(num))

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    outdata_dir = os.path.join(OUT_DIR,str(num),'data')
    outimg_dir = os.path.join(OUT_DIR,str(num),'img')
    if not os.path.exists(outdata_dir):
        os.mkdir(outdata_dir)
    if not os.path.exists(outimg_dir):
        os.mkdir(outimg_dir)

    bbox_list = np.zeros((len(os.listdir(indir)),4))
    layer_list = np.zeros(len(os.listdir(indir)))

    wshed_dict = {}
    
    for i,name in enumerate(os.listdir(indir)):
        wshed = np.load(os.path.join(indir,name))
        wshed = (wshed>0).astype(np.int)
        wshed_dict[name] = wshed

        digits = [int(s) for s in name.split('-') if s.isdigit()]

        bbox_list[i,0] = digits[0]
        bbox_list[i,1] = digits[0]+wshed.shape[1]
        bbox_list[i,2] = digits[1]
        bbox_list[i,3] = digits[1]+wshed.shape[0]

        layer_list[i] = digits[2]
    
    final_bbox = np.zeros(6)
    final_bbox[0] = np.min(bbox_list[:,0])
    final_bbox[1] = np.max(bbox_list[:,1])
    final_bbox[2] = np.min(bbox_list[:,2])
    final_bbox[3] = np.max(bbox_list[:,3])
    final_bbox[4] = np.min(layer_list)
    final_bbox[5] = np.max(layer_list)

    for i,name in enumerate(os.listdir(indir)):
        # name = 'x-11352-y-23144-z-2189-count-52.npy'
        left_pad = int(bbox_list[i,0] - final_bbox[0])
        right_pad = int(- bbox_list[i,1] + final_bbox[1])
        top_pad = int(bbox_list[i,2] - final_bbox[2])
        bottom_pad = int(- bbox_list[i,3] + final_bbox[3])

        result_img = np.pad(wshed_dict[name],((top_pad,bottom_pad),(left_pad,right_pad)),'constant')
        print(result_img.shape)

        result_name = 'x-%s-y-%s-z-%s.png'%(int(final_bbox[0]),int(final_bbox[2]),int(layer_list[i]))

        cv2.imwrite(os.path.join(outdata_dir,result_name),result_img)

        img_name = 'x-%s-y-%s-z-%s.png'%(int(final_bbox[0]),int(final_bbox[2]),int(layer_list[i]))
        plt.figure()
        plt.imshow(result_img)
        plt.savefig(os.path.join(outimg_dir,img_name))
        plt.close()

        print(result_name)

        

    
