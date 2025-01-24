#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from pathlib import Path
import os
import json
import copy 
import itertools

import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from skimage import color

from color_utils import *

basePath = Path(os.getcwd())

# load annotations
year = 2021
cam = 'RGB-1'
date = '20241205'
version = f'{date}-GW'

dataPath = os.path.join(basePath.parent, 'Data/The_Growing_Strawberries')
annotation_file = os.path.join(dataPath, 'annotations', f'{cam}-{year}.json')

with open(annotation_file, 'r') as f:
    data = json.load(f)
det_df = pd.DataFrame(data['annotations'])

gt = []
for anno in enumerate(data['annotations']):
    gt_line = [anno['image_id'], anno['track_id']] + anno['bbox']
    gt.append(gt_line)
detection = np.array(gt)

# load existing progress
if os.path.exists(f'{year}_{cam}_{version}_prog.npy'):
    det_with_fea = np.load(f'{year}_{cam}_{version}_prog.npy')
    ini_frame = det_with_fea[:,0].max()+1
else: 
    det_with_fea = None
    det_with_rgb = None
    det_with_lab = None
    det_with_hsv = None

    ini_frame = 0

to_ten = [-1,-1,-1] # position keeper for MOT dataset format

# start extracting features
print(f'Starting from frame {ini_frame}..')

df1 = pd.DataFrame(data['images'])
df1 = df1.rename(columns={'id':'image_id'})
df2 = pd.DataFrame(data['annotations'])
df2[['x','y','w','h']] = df2['bbox'].to_list()
df_anno = df2.merge(df1[['image_id','file_name']])
df_anno['file_name'] = f'{cam}/'+df_anno['file_name']
df_anno = df_anno[['file_name', 'image_id', 'track_id', 'x', 'y', 'w', 'h']]
img_idx_dict = dict(zip(df_anno['file_name'],df_anno['image_id']))

for img in np.sort(df_anno['file_name'].unique()):
    img_file = os.path.join(dataPath, year, cam, img)

    im = torchvision.io.read_image(img_file)
    assert im is not None
    assert im.shape[0]==3

    # if apply color correction, e.g. gray_world
    # im = gray_world_algorithm(im)

    frame = img_idx_dict[img]
    sel_det = detection[np.where(detection[:,0]==frame)]

    if len(sel_det)>0:
        for i in range(len(sel_det)):
            plg = sel_det[i,2:6]
            if len(plg)>0:
                x,y,w,h = plg
                if w>1 and h>1:
                    bbox_img = sel_middle_per(plg, im, percent=50)
                    color_feature = color_encoder(bbox_img)
                    feature_flattern = list(itertools.chain(*color_feature))
                    feature_flattern.append(1) #1 for bbox det

                    txt_det = list(sel_det[i,:6])
                    txt_det_analysis = txt_det + feature_flattern
                    
                    if det_with_fea is not None:
                        det_with_fea = np.vstack((det_with_fea, np.array(txt_det_analysis).reshape(1,-1)))
                    else:
                        det_with_fea = np.array(txt_det_analysis).reshape(1,-1)
                else:
                    print('too small segment: ', sel_det['image_id'][i], sel_det['track_id'][i])
            
            else:
                print('no segment: ', sel_det['image_id'][i], sel_det['track_id'][i])

    if frame%50==0:
        np.save(f'{year}_{cam}_{version}_prog.npy', det_with_fea)
        print(f'Save {year}_{cam}_{version}_prog.npy progress: frame {frame}.')

np.save(f'{year}_{cam}_{version}_prog.npy', det_with_fea)
print(f'Final save {year}_{cam}_{version}_prog.npy progress: frame {frame}.')

if frame == int(detection[:,0].max()):
    np.save(f'{year}_{cam}_{version}_final.npy', det_with_fea)    
    os.remove(f'{year}_{cam}_{version}_prog.npy')
    print(f'Progress {cam}-{version} finishes at: frame {frame}.')