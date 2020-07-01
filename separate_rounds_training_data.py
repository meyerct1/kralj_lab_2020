#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:06:00 2020

@author: meyerct6
"""


import glob
import pandas as pd
from shutil import copyfile
import os

fil = '/media/hd1/unet_model_training_data_cm/unet-master/data_gfp/unet_data/imKey_little-delta_1_total-time_15.txt'
df = pd.read_csv(fil,sep='\t')
base_dir = '/media/hd1/unet_model_training_data_cm/unet-master/data_gfp/unet_data/%s_little-delta_1_total-time_15/'

dest_base_dir = '/media/hd1/unet_model_training_data_cm/unet-master/data_gfp/unet_data/round_%i_%s_little-delta_1_total-time_15/'


for i in df.index:
    print(str(i) + ' of ' + str(len(df)))
    r = int(df.loc[i,'Im1'].split('/')[-4].split(' ')[1])
    dest_base = dest_base_dir%(r,df.loc[i,'Test/Train'])
    dest_im = dest_base + 'image/' + str(df.loc[i,'Number']) + '.png'
    dest_lab = dest_base + 'label/' + str(df.loc[i,'Number']) + '.png'
    source_base = base_dir%(df.loc[i,'Test/Train'])
    source_im = source_base + 'image/' + str(df.loc[i,'Number']) + '.png'
    source_lab = source_base + 'label/' + str(df.loc[i,'Number']) + '.png'
    os.makedirs(os.path.dirname(dest_im),exist_ok=True)
    os.makedirs(os.path.dirname(dest_lab),exist_ok=True)
    copyfile(source_im,dest_im)
    copyfile(source_lab,dest_lab)
