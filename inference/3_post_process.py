#----------------------------------------------------------------------------
# IMPORTING MODULES
#----------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import os
import cv2
import numpy as np
from numpy import ma
import h5py as h

from scipy import ndimage as ndi
from skimage.morphology import watershed
from scipy.ndimage import label as label_scipy
from skimage.feature import peak_local_max
# from scipy.misc import imsave

from utils.helper import *
from matplotlib import pyplot as plt
from config import *

#----------------------------------------------------------------------------
# ENERGY MAP
#----------------------------------------------------------------------------
for num in range(19):
# for num in range(3,4):
    # num=12
    SYNAPSE_NUM = num
    ENERGY_DIR = os.path.join(OUTPUT_DIR,'/mask_result',str(SYNAPSE_NUM),'pred')
    MASK_DIR = os.path.join(OUTPUT_DIR,'mask_result',str(SYNAPSE_NUM),'mask')
    OUT_DIR = os.path.join(OUTPUT_DIR,'/wshed_result')
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    if not os.path.exists(os.path.join(OUT_DIR,str(SYNAPSE_NUM))):
        os.mkdir(os.path.join(OUT_DIR,str(SYNAPSE_NUM)))
    WSHED_DIR = os.path.join(OUT_DIR,str(SYNAPSE_NUM),'data')
    if not os.path.exists(WSHED_DIR):
        os.mkdir(WSHED_DIR)
    IMG_DIR = os.path.join(OUT_DIR,str(SYNAPSE_NUM),'img')
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)

    COUNT_THRESH1 = 10
    COUNT_THRESH2 = 0
    BG_ENERGY_THRESH = 20
    MARKER_ENERGY_THRESH = 40

    VESICLE_AREA = 100

    # energy = np.array(h.File('./inputs/test_energy.h5', 'r')['main'])[0]
    marker_size = []

    for name in os.listdir(MASK_DIR)[:]:
        # name = 'x-11512-y-23152-z-2193-count-56.png'
        print(num,name)
        mask = cv2.imread(os.path.join(MASK_DIR,name),0)

        digits = [int(s) for s in name.split('-') if s.isdigit()]
        name = 'x-%s-y-%s-z-%s.npy'%tuple(digits[:3])

        energy = np.load(os.path.join(ENERGY_DIR,name))
        energy = energy/np.max(energy)*255
        energy = cv2.resize(energy,mask.shape)
        energy = energy[np.newaxis,:,:]
        energy.shape

        #----------------------------------------------------------------------------
        # CONNECTED COMPONENTS
        #----------------------------------------------------------------------------

        # Apply connected components to get initial segmentation map. 

        seg = get_seg(energy, None, BG_ENERGY_THRESH) 

        # Remove small segmentation labels (with voxel count < COUNT_THRESH)

        nlabels, count = np.unique(seg, return_counts=True)

        indices = np.argsort(count)
        nlabels = nlabels[indices]
        count = count[indices]

        least_index = np.where(count >= COUNT_THRESH1)[0][0] 

        count = count[least_index:]
        nlabels = nlabels[least_index:]

        rl = np.arange(seg.max() + 1).astype(seg.dtype)

        for i in range(seg.max() + 1):
            if i not in nlabels:
                rl[i] = 0

        seg = rl[seg]

        # # Save initial segmentation

        # f = h.File(f'./inputs/test_seg_cc.h5', 'w')
        # f.create_dataset('main', data=seg)
        # f.close()

        #----------------------------------------------------------------------------
        # WATERSHED
        #----------------------------------------------------------------------------

        # energy = np.array(h.File('./inputs/test_energy.h5', 'r')['main'])[0].astype(np.float32)

        threshold = MARKER_ENERGY_THRESH

        # Extract markers from energy map

        energy_thres = energy - threshold

        markers_unlabelled = (energy_thres > 0).astype(int)

        # Label markers for watershed

        markers, ncomponents = label_scipy(markers_unlabelled)

        # Remove small markers, to prevent oversegmentation

        labels_d, count_d = np.unique(markers, return_counts=True) 

        marker_size += list(count_d[1:])

        rl = np.arange(markers.max() + 1).astype(markers.dtype)

        pixel_threshold = COUNT_THRESH2


        for i in range(len(labels_d)):
            if count_d[i] < pixel_threshold:
                rl[labels_d[i]] = 0

        markers = rl[markers]

        # Mask for watershed from CC output

        mask = (seg > 0).astype(int) 

        # Watershed with markers and mask 

        labels = watershed(-energy, mask=mask, markers=markers) 

        # # show contrast results
        # plt.subplot(1,4,1)
        # plt.imshow(markers_unlabelled[0,:,:])
        # plt.title('original seed')
        # plt.subplot(1,4,2)
        # plt.imshow(labels[0])
        # plt.title('original seg')

        # --------------------------------------------------------------------------------------
        # iterative watershed
        # --------------------------------------------------------------------------------------

        flags = needIteration(labels,VESICLE_AREA)
        erosion_map = np.zeros(labels.shape)
        if np.sum(flags)>0:
            iter_loc = np.where(flags>0)
            for i in iter_loc[0]:
                if i==0:
                    continue
                this_mask = (labels==i)
                this_heat = this_mask*energy
                this_max = peak_local_max(this_heat[0],min_distance=4,num_peaks_per_label=1)
                value_list = []
                max_pos_list = []
                for j in range(this_max.shape[0]):
                    tmp_value = this_heat[0][this_max[j,0],this_max[j,1]]
                    if tmp_value not in value_list:
                        value_list.append(tmp_value)
                        max_pos_list.append(this_max[j,:])
                    # elif np.linalg.norm()
                this_max = np.array(max_pos_list)

                if this_max.shape[0] > 1:
                    erosion_map[this_mask] = 1
                    markers_unlabelled[this_mask] = 0
                    markers_unlabelled[0][this_max[:,0],this_max[:,1]] = 1
                    # Label markers for watershed
                    markers, ncomponents = label_scipy(markers_unlabelled)  

                    # if i==47:
                    #     print("here")            
                    #     plt.subplot(1,3,1)
                    #     plt.imshow(labels[0])
                    #     plt.subplot(1,3,2)
                    #     plt.imshow(this_mask[0])
                    #     plt.subplot(1,3,3)
                    #     plt.imshow(this_heat[0])
                    #     plt.plot(this_max[:,1],this_max[:,0],'ro')
                    #     plt.show()

        seg_label, orig_count = label_scipy(seg[0])

        for i in range(1,orig_count+1):
            this_mask = seg_label==i
            this_region = this_mask*labels
            this_seed = np.unique(this_region)
            this_seed_num = np.sum(this_seed>0)
            if this_seed_num>1:
                erosion_map[0,this_mask] = 1
        
        labels = watershed(-energy, mask=mask, markers=markers)

        for i in range(1,np.max(labels)+1):
            for j in range(2):
                this_mask = labels==i
                this_flag = this_mask*erosion_map
                if np.sum(this_flag) > 40:
                    # kernel = np.ones((3,3),np.uint8)
                    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
                    new_mask = cv2.erode(this_mask[0].astype(np.uint8),kernel,iterations = 1)
                    labels[this_mask] = 0
                    labels[0,new_mask.astype(np.bool)] = 1*i
        

        # # show contrast result.  
        # plt.subplot(1,4,3)
        # plt.imshow(markers_unlabelled[0])
        # plt.title('final seed')
        # plt.subplot(1,4,4)
        # plt.imshow(labels[0,:,:])
        # plt.title('final seg')
        # plt.show()

        # save watershed results
        wshed_name = 'x-%s-y-%s-z-%s-count-%s.npy'%tuple(digits[:3]+[int(np.max(labels)),])
        labels = cv2.resize(labels[0].astype(np.uint8),(labels.shape[1],int(labels.shape[2]/2)),interpolation=cv2.INTER_NEAREST)
        # labels = labels[0]
        np.save(os.path.join(WSHED_DIR,wshed_name),labels)
        img_name = 'x-%s-y-%s-z-%s-count-%s.png'%tuple(digits[:3]+[int(np.max(labels)),])
        plt.figure()
        plt.imshow(labels)
        plt.savefig(os.path.join(IMG_DIR,img_name))
        plt.close()

        

        # Save Watershed segmentation map

        # f = h.File(f'./inputs/test_seg_{threshold}.h5', 'w')
        # f.create_dataset('main', data=labels)
        # f.close()

        # Get watershed labels for Neuroglancer

        np.unique(labels) 

        # # show the result  

        # plt.subplot(1,4,1)
        # plt.imshow(energy[0,:,:])
        # # plt.imshow(erosion_map[0,:,:])
        # plt.title('origin img')
        # plt.subplot(1,4,2)
        # plt.imshow(seg[0])
        # plt.title('CC seg')
        # plt.subplot(1,4,3)
        # plt.imshow(markers_unlabelled[0])
        # plt.title('marker candidates')
        # plt.subplot(1,4,4)
        # plt.imshow(labels[:,:])
        # plt.title('final seg')
        # plt.show()

        # labels = cv2.resize(labels[0].astype(np.uint8),(labels.shape[2],int(labels.shape[1]/2)))



        #----------------------------------------------------------------------------
        # VISUALIZATION
        #----------------------------------------------------------------------------

        # from neuroG import NeuroG

        # ng = NeuroG(port=8891)


        # ng.addLayer('inputs/test_volm.h5', 'h5py', name="Input Image")
        # ng.addLayer('inputs/test_energy.h5', 'h5py', name="Output")
        # ng.addLayer(f'inputs/test_seg_{threshold}.h5', 'h5py', name="Segmentation", isLabel=True)
        # ng.addLayer(f'inputs/test_seg_cc.h5', 'h5py', name="Seg CC", isLabel=True)

        # ng.viewer


        #----------------------------------------------------------------------------
        # SAVE OUTPUT TO PNG
        #----------------------------------------------------------------------------

        # def seg2Vast(seg):                                                                                   
        #     return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)

        # labels_wshed = np.array(h.File('./inputs/test_seg_150.h5', 'r')['main'])

        # for i in range(labels_wshed.shape[0]):
        #     png = seg2Vast(labels_wshed[i])
        #     # save png to file
        #     imsave(f'./slice20/{i}.png', png)

    index,count = np.unique(marker_size,return_counts=True)

    if len(count) > 0:

        hist,bins = np.histogram(marker_size,bins=len(index))

        hist_avg = hist/np.sum(hist)

        # plt.hist(marker_size,bins=len(index),color = "orange", ec="orange")
        # # plt.plot(bins[1:],hist)
        # plt.plot(bins[1:],np.cumsum(hist_avg)*np.max(hist))
        # plt.title("num  v.s.  CC-area")
        # plt.xlabel("CC Area")
        # plt.ylabel("CC num")

        weighted_avg = np.sum(index*count/np.sum(count))
        print("weighted avg:",weighted_avg)
        # plt.show()
