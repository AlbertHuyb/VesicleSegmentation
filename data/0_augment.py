import os
import cv2
import random
import string
import skimage as sk
from config import *

def rand_string_gen(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for x in range(size))

def randomCrop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    img = cv2.resize(img,(2*width,2*height))
    mask = cv2.resize(mask,(2*width,2*height))
    return img, mask

def randomRotation(image_array,gt_array,angle):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-angle, angle)
    rows,cols = image_array.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),random_degree,1)
    img = cv2.warpAffine(image_array,M,(cols,rows))
    gt = cv2.warpAffine(gt_array,M,(cols,rows))
    return img,gt

INPUT_DIR = RAW_DATA 
INPUT_IM = os.path.join(INPUT_DIR,'im')
INPUT_GT = os.path.join(INPUT_DIR,'gt')

OUTPUT_DIR = AUG_DIR 
OUTPUT_IM = os.path.join(OUTPUT_DIR,'im')
OUTPUT_GT = os.path.join(OUTPUT_DIR,'gt')

img_list = os.listdir(INPUT_IM)

for i in img_list:
    orig_im = cv2.imread(os.path.join(INPUT_IM,i),0)
    orig_gt = cv2.imread(os.path.join(INPUT_GT,i),0)
    name = rand_string_gen()+'.png'
    orig_im = cv2.resize(orig_im,(1024,768))
    orig_gt = cv2.resize(orig_gt,(1024,768))
    orig_gt[orig_gt>0] = 255
    # cv2.imshow('1',orig_im)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(OUTPUT_IM,name),orig_im)
    cv2.imwrite(os.path.join(OUTPUT_GT,name),orig_gt)
    # crop
    for i in range(5):
        im_temp,gt_temp = randomCrop(orig_im,orig_gt,512,384)
        name = rand_string_gen()+'.png'
        cv2.imwrite(os.path.join(OUTPUT_IM,name),im_temp)
        cv2.imwrite(os.path.join(OUTPUT_GT,name),gt_temp)
    # rotation
    for i in range(3):
        im_temp,gt_temp = randomRotation(orig_im,orig_gt,30)
        name = rand_string_gen()+'.png'
        cv2.imwrite(os.path.join(OUTPUT_IM,name),im_temp)
        cv2.imwrite(os.path.join(OUTPUT_GT,name),gt_temp)

    # vertical
    im = cv2.flip(orig_im,0)
    gt = cv2.flip(orig_gt,0)
    name = rand_string_gen()+'.png'
    cv2.imwrite(os.path.join(OUTPUT_IM,name),im)
    cv2.imwrite(os.path.join(OUTPUT_GT,name),gt)
    # crop
    for i in range(5):
        im_temp,gt_temp = randomCrop(im,gt,512,384)
        name = rand_string_gen()+'.png'
        cv2.imwrite(os.path.join(OUTPUT_IM,name),im_temp)
        cv2.imwrite(os.path.join(OUTPUT_GT,name),gt_temp)
    # rotation
    for i in range(3):
        im_temp,gt_temp = randomRotation(im,gt,30)
        name = rand_string_gen()+'.png'
        cv2.imwrite(os.path.join(OUTPUT_IM,name),im_temp)
        cv2.imwrite(os.path.join(OUTPUT_GT,name),gt_temp)

    # horizontal
    im = cv2.flip(orig_im,1)
    gt = cv2.flip(orig_gt,1)
    name = rand_string_gen()+'.png'
    cv2.imwrite(os.path.join(OUTPUT_IM,name),im)
    cv2.imwrite(os.path.join(OUTPUT_GT,name),gt)
    # crop
    for i in range(5):
        im_temp,gt_temp = randomCrop(im,gt,512,384)
        name = rand_string_gen()+'.png'
        cv2.imwrite(os.path.join(OUTPUT_IM,name),im_temp)
        cv2.imwrite(os.path.join(OUTPUT_GT,name),gt_temp)
    # rotation
    for i in range(3):
        im_temp,gt_temp = randomRotation(im,gt,30)
        name = rand_string_gen()+'.png'
        cv2.imwrite(os.path.join(OUTPUT_IM,name),im_temp)
        cv2.imwrite(os.path.join(OUTPUT_GT,name),gt_temp)

    # central
    im = cv2.flip(orig_im,-1)
    gt = cv2.flip(orig_gt,-1)
    name = rand_string_gen()+'.png'
    cv2.imwrite(os.path.join(OUTPUT_IM,name),im)
    cv2.imwrite(os.path.join(OUTPUT_GT,name),gt)
    # crop
    for i in range(5):
        im_temp,gt_temp = randomCrop(im,gt,512,384)
        name = rand_string_gen()+'.png'
        cv2.imwrite(os.path.join(OUTPUT_IM,name),im_temp)
        cv2.imwrite(os.path.join(OUTPUT_GT,name),gt_temp)
    # rotation
    for i in range(3):
        im_temp,gt_temp = randomRotation(im,gt,30)
        name = rand_string_gen()+'.png'
        cv2.imwrite(os.path.join(OUTPUT_IM,name),im_temp)
        cv2.imwrite(os.path.join(OUTPUT_GT,name),gt_temp)


