import cv2
from skimage.measure import label
import numpy as np
from matplotlib import pyplot as plt

def Gtgenerator(input):
    input = cv2.cvtColor(input,cv2.COLOR_RGB2GRAY)
    label_map = label(input)

    total_num = np.max(label_map)
    regions = []
    centers = np.zeros((total_num,2))
    result = np.zeros(input.shape)

    for n in range(1,total_num+1):
        sub_image = label_map == n
        # print(sub_image.shape)
        mask = np.where(sub_image>0)
        top = np.min(mask[0])
        down = np.max(mask[0])
        left = np.min(mask[1])
        right = np.max(mask[1])
        height = down - top
        width = right - left 

        centers[n-1,0] = int(np.round(top/2+down/2))
        centers[n-1,1] = int(np.round(left/2+right/2))

        regions.append({'center':(centers[n-1,0],centers[n-1,1]),'height':height,'width':width})

        box_size = int(np.round(height/2+width/2))
        this_kernel = cv2.getGaussianKernel(box_size,0.3*((box_size*1.5-1)*0.5 - 1) + 0.8)
        this_kernel = np.kron(this_kernel,np.transpose(this_kernel))

        # result[int(centers[n-1,0]-round(box_size/2)):int(centers[n-1,0]-round(box_size/2))+box_size,int(centers[n-1,1]-round(box_size/2)):int(centers[n-1,1]-round(box_size/2))+box_size] += this_kernel
        this_size = result[top:top+box_size,left:left+box_size].shape
        result[top:top+box_size,left:left+box_size] += this_kernel[:this_size[0],:this_size[1]]

    return result