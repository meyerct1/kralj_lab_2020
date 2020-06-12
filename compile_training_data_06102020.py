#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:55:04 2020

@author: meyerct6
"""


#Import packages for image processing
from AST_unet.image_preprocessing import frame_extraction,image_chop
import os
import glob
from shutil import rmtree, copyfile
import multiprocessing
import pandas as pd

MAX_NUMBER_OF_FRAMES = 60


df= pd.read_csv('compile_training_data.csv')

# #check to make sure files exist
# for i in df.index:
#     fils = glob.glob(df.loc[i,'Source'])
#     if len(fils)==0:
#         print(i)

a_fils = list(df['Source'])
regexp = [i.split('/')[-1][1:] for i in a_fils]
a_fils = ['/'.join(i.split('/')[:-1]) for i in a_fils]
b_fils = [i+os.sep+'extracted_frames'+os.sep for i in df['Location']]
c_fils = [i+os.sep+'ready_data'+os.sep for i in df['Location']]

#Processing function
def image_preprocessing(a,b,c,r):
    bol = True
    while bol:
        try:
            if not os.path.isdir(c):
                for directory in [b,c]:
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
            
                frame_extraction(a,b,regexp=r,trim=True,max_frame=MAX_NUMBER_OF_FRAMES)       # extracts frames to folder b
                for f in os.listdir(b):
                    if f.endswith('.avi'):
                        directory = os.path.join(c,f) + os.sep
                        if os.path.isdir(directory):
                            rmtree(directory)
                        os.makedirs(directory)
                        frames = glob.glob(os.path.join(b,f,'*.png'))
                        for fr in frames:
                            image_chop(fr, directory, 256)         # Takes subsections of images and saves them to the "ready" folder
                #Copy the segmentation settings file
                copyfile('/media/hd1/unet_model_search_05082020/unet-master/data_gfp/All resistant/ready_data/Settings.csv',c+ 'Settings.csv')
        
            else:
                print('Skipping ' + a)

            bol = False
        except:
            print('Error, retrying ' + a)
            bol = True


for (a,b,c,r) in zip(a_fils,b_fils,c_fils,regexp):
    image_preprocessing(a,b,c,r)

# pool = multiprocessing.Pool(processes=20)
# inputs = zip(a_fils,b_fils,c_fils,regexp)
# outputs = pool.starmap(image_preprocessing, inputs)

