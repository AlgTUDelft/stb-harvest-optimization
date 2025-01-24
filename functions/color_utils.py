import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from skimage import color
import copy
from torchvision import transforms

def xywh2xyxy(bbox):
    new_bbox = copy.copy(bbox)
    new_bbox[2] = new_bbox[2] + new_bbox[0]
    new_bbox[3] = new_bbox[3] + new_bbox[1]
    return new_bbox

def get_segment(imArray, mask, frame_id, tar_shape=[200, 200, 3]):
    box_xywh = copy.copy(mask)
    coImArray = copy.copy(imArray)

    if box_xywh.min()<0:
        print(f'Negative value of bbox at {frame_id}.')
        box_xywh[np.where(box_xywh<0)]=0
    if box_xywh[1]+box_xywh[3]>coImArray.shape[0]:
        print(f'Bottom over border at {frame_id}.')
        box_xywh[3] = coImArray.shape[0]-box_xywh[1]-1
    if box_xywh[0]+box_xywh[2]>coImArray.shape[1]:
        print(f'Right over border at {frame_id}.')
        box_xywh[2] = coImArray.shape[1]-box_xywh[0]-1
    seg = coImArray[box_xywh[1]:box_xywh[1]+box_xywh[3], box_xywh[0]:box_xywh[0]+box_xywh[2], :]

    diff_h = (tar_shape[0] - box_xywh[3]+1)//2 #somewhere in 3000
    diff_w = (tar_shape[1] - box_xywh[2]+1)//2 #somewhere in 4000
    if diff_h>0:
        compensate_row = np.zeros(shape=(diff_h, box_xywh[2], 3))
        compensate_row_b = np.zeros(shape=(tar_shape[1] - box_xywh[3] - diff_h, box_xywh[2], 3))
        seg = np.vstack((compensate_row, seg))
        seg = np.vstack((seg,compensate_row_b))
    else:
        seg = seg[-diff_h:-diff_h+tar_shape[1],:,:]

    if diff_w>0:    
        compensate_col = np.zeros(shape=(tar_shape[0], diff_w, 3))
        compensate_col_r = np.zeros(shape=(tar_shape[0], tar_shape[0] - box_xywh[2] - diff_w, 3))
        seg = np.hstack((compensate_col, seg))
        seg = np.hstack((seg, compensate_col_r))
    else:
        seg = seg[:,-diff_w:-diff_w+tar_shape[0],:]

    return seg

def uni_size(seg, tar_shape):
    transform = transforms.Compose(
    [
        transforms.Pad((tar_shape[0]//2,tar_shape[1]//2,tar_shape[0]//2,tar_shape[1]//2)),
        transforms.CenterCrop((tar_shape[0],tar_shape[1]))
    ]
)
    seg = transform(seg)
    return seg

def sel_middle_per(bbox, img, percent=100):
    x,y,w,h = bbox
    x = np.max([0,x])
    y = np.max([0,y])
    alpha = (1 - math.sqrt(percent/100))/2

    x1 = max(int(x+alpha*w), 0)
    x2 = min(int(x+(1-alpha)*w), img.shape[2])
    y1 = max(int(y+alpha*h), 0)
    y2 = min(int(y+(1-alpha)*h), img.shape[2])
    
    return img[:, y1:y2, x1:x2]/255

def color_dominance(channel_hist, area, dominance_range=5):
    # Find the bin with the highest count for each color channel
    color_max_bin = np.argmax(channel_hist)

    # Compute the ratio of the dominant color (±range) for each color channel
    color_ratio = np.sum(channel_hist[color_max_bin-dominance_range:color_max_bin+dominance_range]) / area * 100

    return color_ratio

def dominate_color(img_arr):
    assert img_arr.shape[0]==3
    area = len(np.where(img_arr.sum(axis=0)!=0)[0])
    r_hist, g_hist, b_hist = color_hist(img_arr)

    ratio = []
    for hist in [r_hist, g_hist, b_hist]:
        color_ratio = color_dominance(hist, area)
        ratio.append(color_ratio)

    # Compute the overall ratio of the dominant color
    # /3 is just an approximate replace the sum calculation
    dominant_ratio = sum(ratio)/3

    ratio.append(dominant_ratio)

    return ratio

def color_features(img_data):
    assert len(img_data.shape)==2, 'img is not flattern'
    assert img_data.shape[0]==3, 'wrong img shape'

    average = tuple(np.mean(img_data, axis=1))
    dominant = tuple(np.median(img_data, axis=1).astype(np.uint8))
    dominance = dominate_color(img_data)
    return average, dominant, dominance

def color_encoder(seg, mask=[]):
    if type(seg)==torch.Tensor:
        seg = seg.detach().numpy()

    if len(mask)>0:
        mask_bool = np.squeeze(mask).numpy().astype(bool)
        img_data = seg[:,mask_bool].reshape((3,-1))
    
    else:
        img_data = seg.reshape((3,-1))

    # Calculate the dominant color (i.e., the most common RGB value)
    average_rgb, dominant_rgb, dominance_rgb = color_features(img_data)

    img_data_lab = color.rgb2lab(img_data.transpose()).transpose()
    average_lab, dominant_lab, dominance_lab = color_features(img_data_lab)

    img_data_hsv = color.rgb2hsv(img_data.transpose()).transpose()
    average_hsv, dominant_hsv, dominance_hsv = color_features(img_data_hsv)

    return dominant_rgb, average_rgb, dominance_rgb, dominant_lab, average_lab, dominance_lab,  dominant_hsv, average_hsv, dominance_hsv


def color_hist(img_arr):
    assert img_arr.shape[0]==3

    r_hist = np.histogram(img_arr[0,:], bins=256, range=(0,255))[0]
    g_hist = np.histogram(img_arr[1,:], bins=256, range=(0,255))[0]
    b_hist = np.histogram(img_arr[2,:], bins=256, range=(0,255))[0]

    assert np.isnan(r_hist).sum()==0

    return r_hist, g_hist, b_hist

def gray_world_algorithm(im):
    image = im.cpu().detach().numpy()

    avg_r = np.mean(image[0, :, :])
    avg_g = np.mean(image[1, :, :])
    avg_b = np.mean(image[2, :, :])
    
    avg_gray = (avg_r + avg_g + avg_b) / 3
    scale_r = avg_gray / avg_r
    scale_g = avg_gray / avg_g
    scale_b = avg_gray / avg_b
    
    image[0, :, :] = np.clip(image[0, :, :] * scale_r, 0, 255)
    image[1, :, :] = np.clip(image[1, :, :] * scale_g, 0, 255)
    image[2, :, :] = np.clip(image[2, :, :] * scale_b, 0, 255)
    
    image = image.astype(np.uint8)
    
    return image