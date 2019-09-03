import os
import cv2
import numpy as np
from utils import *
from config import *
# from matplotlib import pyplot as plt

SOURCE_DIR = AUG_DIR
IMAGE_DIR = os.path.join(SOURCE_DIR,'im')
LABEL_DIR = os.path.join(SOURCE_DIR,'gt')
# DES_DIR = '/home/huyb/EM/CrowdCounting/ProcessedData/Vesicle_pad' 
DES_DIR = DATA_DIR

TOTAL_NUM = 360

TRAIN_NUM = 300 # int(TOTAL_NUM*0.4) 
VALID_NUM = 60 # int(TOTAL_NUM*0.4) 
TEST_NUM =  0 # TOTAL_NUM-TRAIN_NUM-VALID_NUM

STANDARD_SIZE = (1024,768) # width 1024 height 768

# data_list = open(os.path.join(SOURCE_DIR,'vesicle_list.txt')).readlines()
data_list = os.listdir(IMAGE_DIR)

for num_counter,name in enumerate(data_list):
    print(num_counter,name)

    # name = name[:-1]

    img_from = os.path.join(IMAGE_DIR,name)
    seg_from = os.path.join(LABEL_DIR,name)

    if num_counter < TRAIN_NUM:
        destination = os.path.join(DES_DIR,'train_data')  
        savenum = num_counter+1     
    elif num_counter < TRAIN_NUM+VALID_NUM:
        destination = os.path.join(DES_DIR,'valid_data')
        savenum = num_counter-TRAIN_NUM+1
    elif num_counter < TRAIN_NUM+VALID_NUM+TEST_NUM:
        destination = os.path.join(DES_DIR,'test_data')
        savenum = num_counter-TRAIN_NUM-VALID_NUM+1
    else:
        exit()
    
    print(destination,savenum)
    
    img_des = os.path.join(destination,'img')
    den_des = os.path.join(destination,'den')
    seg_des = os.path.join(destination,'seg')

    orig_img = cv2.imread(img_from)
    temp_size = orig_img.shape
    save_img = cv2.resize(orig_img,STANDARD_SIZE)

    seg_img = cv2.imread(seg_from)
    
    seg_img[seg_img>0] = 1

    save_seg = cv2.resize(seg_img,STANDARD_SIZE)

    save_gt = save_seg
    # plt.subplot(1,2,1)
    # plt.imshow(save_seg.astype('float'))
    # plt.subplot(1,2,2)
    # plt.imshow(save_gt)
    # plt.show()
    # save_seg = cv2.copyMakeBorder(seg_img,top_pad,bottom_pad,left_pad,right_pad,cv2.BORDER_CONSTANT,value=0)

    #### write in the directories
    cv2.imwrite(os.path.join(img_des,str(savenum)+'.png'),save_img)
    cv2.imwrite(os.path.join(seg_des,str(savenum)+'.png'),save_seg*255)
    np.savetxt(os.path.join(den_des,str(savenum)+'.csv'),cv2.cvtColor(save_gt,cv2.COLOR_BGR2GRAY),delimiter=',',fmt='%.2f')
    



