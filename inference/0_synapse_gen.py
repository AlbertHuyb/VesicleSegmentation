import os,sys
import h5py
import numpy as np
import cv2
import json
from skimage.measure import label
from matplotlib import pyplot as plt
from utils.T_util import bfly
from config import *

def readh5(filename, datasetname='main', rr=[1,1,1]):
    fid=h5py.File(filename,'r')
    if isinstance(datasetname, (list,)):
        out = [None] *len(datasetname)
        for i,dd in enumerate(datasetname):
            sz = len(fid[dd].shape)
            if sz==1:
                out[i] = np.array(fid[dd][::rr[0]])
            elif sz==2:
                out[i] = np.array(fid[dd][::rr[0],::rr[1]])
            elif sz==3:
                out[i] = np.array(fid[dd][::rr[0],::rr[1],::rr[2]])
            else:
                out[i] = np.array(fid[dd])
    else:                                                                       
        sz = len(fid[datasetname].shape)
        if sz==1:
            out = np.array(fid[datasetname][::rr[0]])
        elif sz==2:
            out = np.array(fid[datasetname][::rr[0],::rr[1]])
        elif sz==3:
            out = np.array(fid[datasetname][::rr[0],::rr[1],::rr[2]])
        else:
            out = np.array(fid[datasetname])
    return out

def GetBbox(seg, do_count=False):
    dim = len(seg.shape)
    a=np.where(seg>0)
    if len(a)==0:
        return [-1]*dim*2
    out=[]
    for i in range(dim):
        out+=[a[i].min(), a[i].max()]
    if do_count:
        out+=[len(a[0])]
    return out

# problem: x-y transposed...

BOX_SIZE = [256,256,30]
# OUTPUT_DIR = '/mnt/pfister_lab2/yubin/vesiclesNew/vesicle_18_track_1gpu'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# fn = '/mnt/pfister_lab2/donglai/data/JWR/JWR_local-butterfly_dw.json'
fn = SEG_NAME
pos = [5000,5000,500] #x,y,z
psz = [1024,1024,10] # x,y,z
jn = json.load(open(fn))
sn = jn['sections']
sn = ['/mnt/pfister_lab2/'+x[20:] for x in sn] #  coxfs to hp03
tile_sz = [jn['dimensions']['height'],jn['dimensions']['width']]

# mask_dir = '/mnt/pfister_lab2/vcg_connectomics/JWR15/axon_32nm/cell53/'
mask_dir = MASK_DIR

# 4x4x30nm
# syn_pos = np.loadtxt('/home/huyb/EM/VesicleSeg/cell53_syn.txt').astype(int)
syn_pos = np.loadtxt(LOCATION_FILE).astype(int)
# 128x128x120nm
# syn_pos = syn_pos/[32,32,4]

full_outdir = os.path.join(OUTPUT_DIR,'full')
if not os.path.exists(full_outdir):
    os.mkdir(full_outdir)
part_outdir = os.path.join(OUTPUT_DIR,'part')
if not os.path.exists(part_outdir):
    os.mkdir(part_outdir)
mask_outdir = os.path.join(OUTPUT_DIR,'mask')
if not os.path.exists(mask_outdir):
    os.mkdir(mask_outdir)

for i in range(syn_pos.shape[0]):
    # i=12
    this_outdir = os.path.join(full_outdir,str(i))
    if not os.path.exists(this_outdir):
        os.mkdir(this_outdir)
    this_partdir = os.path.join(part_outdir,str(i))
    if not os.path.exists(this_partdir):
        os.mkdir(this_partdir)
    this_maskdir = os.path.join(mask_outdir,str(i))
    if not os.path.exists(this_maskdir):
        os.mkdir(this_maskdir)

    orig_pos = syn_pos[i].astype('int')[::-1]
    pos = orig_pos/[4,32,32]
    # print(pos)

    left = int(pos[2]-BOX_SIZE[0]/2)
    right = int(left+BOX_SIZE[0])
    top = int(pos[1]-BOX_SIZE[1]/2)
    down = int(top+BOX_SIZE[1])

    for layer in [0]+range(int(-BOX_SIZE[2]/2),int(BOX_SIZE[2]/2)):
        print("layer:",layer+orig_pos[0])
        # if i != 0:
        #     break
        mask = cv2.imread(os.path.join(mask_dir,'_s%04d.png'%(layer+orig_pos[0])),0)

        mask = (mask>=1).astype(np.uint8)
        mask_x = orig_pos[1]/8
        mask_y = orig_pos[2]/8

        mask_top = max(0,mask_x-100)
        mask_down = min(mask_top+200,mask.shape[0])
        mask_left = max(0,mask_y-100)
        mask_right = min(mask_left+200,mask.shape[0])

        mask_patch = mask[mask_top:mask_down,mask_left:mask_right]
        label_map = label(mask_patch)
        this_label = label_map[mask_x-mask_top,mask_y-mask_left]

        this_cc = label_map==this_label
        if layer == 0:
            middle_cc = this_cc

        min_dis = 99999
        for i in range(1,np.max(label_map)+1):
            temp_bb = GetBbox(label_map==i)
            this_center = np.array([temp_bb[0]/2+temp_bb[1]/2,temp_bb[2]/2+temp_bb[3]/2])
            this_dis = np.linalg.norm(this_center-np.array([mask_x-mask_top,mask_y-mask_left]))
            if this_dis < min_dis:
                min_dis = this_dis
                final_mask = label_map==i
        this_cc = (final_mask + middle_cc)>0

        if np.sum(this_cc)==0:
            continue
        bb = GetBbox(this_cc) 
        # import pdb; pdb.set_trace()
        im = bfly(sn, (bb[2]+mask_left)*8,(bb[3]+mask_left)*8, \
                (bb[0]+mask_top)*8,(bb[1]+mask_top)*8, \
                layer+orig_pos[0],layer+orig_pos[0]+1, tile_sz, np.uint8)  

        # print(im.shape)
        this_img = im[0,:,:]
        # plt.imshow(this_img)
        # plt.show()
        this_mask = mask_patch[bb[0]:bb[1],bb[2]:bb[3]] # keep the mask on this layer
        mask_img = this_img*cv2.resize(this_mask,(8*this_mask.shape[1],8*this_mask.shape[0]))
        this_img = cv2.resize(this_img,(this_img.shape[1],this_img.shape[0]*2))
        mask_img = cv2.resize(mask_img,(mask_img.shape[1],mask_img.shape[0]*2))
        # if layer>=-30:
        #     plt.imshow(mask_img)
        #     plt.show()
        # this_img = 255*seg[left:right,top:down,layer+pos[0]].astype('uint8')
        # this_name = 'pos_'+str(pos)+'_layer_'+str(layer)
        this_name = 'x-%s-y-%s-z-%s'%((bb[2]+mask_left)*8,(bb[0]+mask_top)*8,layer+orig_pos[0])

        cv2.imwrite(os.path.join(this_outdir,this_name+'.png'),this_img)
        cv2.imwrite(os.path.join(this_partdir,this_name+'.png'),mask_img)
        cv2.imwrite(os.path.join(this_maskdir,this_name+'.png'),255*cv2.resize(this_mask,(8*this_mask.shape[1],16*this_mask.shape[0])))

        # if layer == 0:
        #     orig_tmp = bfly(sn, 0,26624, \
        #         0,26624, \
        #         layer+orig_pos[0],layer+orig_pos[0]+1, tile_sz, np.uint8)
        #     orig_tmp = orig_tmp[0,:,:]

        #     mask = cv2.imread(os.path.join(mask_dir,'_s%04d.png'%(layer+orig_pos[0])),0)

        #     mask = (mask>=1).astype(np.uint8)
        #     mask_x = orig_pos[1]/8
        #     mask_y = orig_pos[2]/8

        #     mask_top = max(0,mask_x-100)
        #     mask_down = min(mask_top+200,mask.shape[0])
        #     mask_left = max(0,mask_y-100)
        #     mask_right = min(mask_left+200,mask.shape[0])

        #     mask_patch = mask[mask_top:mask_down,mask_left:mask_right]
        #     label_map = label(mask_patch)
        #     this_label = label_map[mask_x-mask_top,mask_y-mask_left]
        #     this_cc = label_map==this_label
        #     bb = GetBbox(this_cc)
        #     img_patch = orig_tmp[mask_top*8:mask_down*8,mask_left*8:mask_right*8]
        #     this_img = img_patch[bb[0]*8:bb[1]*8,bb[2]*8:bb[3]*8]
        #     # mask_img = img_patch*cv2.resize(mask_patch,(8*mask_patch.shape[1],8*mask_patch.shape[0]))
            
        #     this_img = cv2.resize(this_img,(this_img.shape[1],this_img.shape[0]*2))
        #     plt.imshow(this_img)
        #     plt.show()
        #     # cv2.imwrite(os.path.join(this_outdir,this_name+'.png'),this_img)
