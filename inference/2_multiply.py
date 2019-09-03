import cv2
import os
from skimage.measure import label
import numpy as np
from config import *

ROOT_FOLDER = OUTPUT_DIR
IN_DIR = os.path.join(ROOT_FOLDER,'result')
# OUT_DIR = '/mnt/pfister_lab2/yubin/vesiclesNew/vesicle_18/mask_result'
OUT_DIR = os.path.join(ROOT_FOLDER,'mask_result')

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

directory_list = [dirs for _,dirs,files in os.walk(IN_DIR)]

for i in directory_list[0]:
    print(i)
    indir = os.path.join(IN_DIR,i)
    outdir = os.path.join(OUT_DIR,i)
    heatdir = os.path.join(outdir,'pred')
    maskdir = os.path.join(outdir,'mask')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(heatdir):
        os.mkdir(heatdir)
    if not os.path.exists(maskdir):
        os.mkdir(maskdir)

    orig_dir = os.path.join(indir,'mask')
    mask_dir = os.path.join(ROOT_FOLDER,'mask',i)
    pred_dir = os.path.join(ROOT_FOLDER,'result',i,'pred')

    for j in os.listdir(orig_dir):
        digits = [int(s) for s in j.split('-') if s.isdigit()]
        orig_img = cv2.imread(os.path.join(orig_dir,j),0)
        orig_heat = np.load(os.path.join(pred_dir,'x-%s-y-%s-z-%s.npy'%tuple(digits[:3])))
        mask = cv2.imread(os.path.join(mask_dir,'x-%s-y-%s-z-%s.png'%tuple(digits[:3])),0)
        out_img = orig_img*mask
        out_heat = orig_heat*mask

        label_map = label(out_img)
        num = np.max(label_map)
        cv2.imwrite(os.path.join(maskdir,'x-%s-y-%s-z-%s-count-%s.png'%(tuple(digits[:3])+(num,))),out_img*255)
        np.save(os.path.join(heatdir,'x-%s-y-%s-z-%s.npy'%(tuple(digits[:3]))),out_heat)
