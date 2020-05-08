#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:57:42 2020

@author: meyerct6
"""

#Import packages for image processing
from mlAbRes.image_preprocessing import frame_extraction,diff_imager,randomize_names,image_chop
import os
import glob
from shutil import rmtree
import multiprocessing

MAX_NUMBER_OF_FRAMES = 240
#Processing function
def image_preprocessing(a,b,c):
    for directory in [b,c]:
        if os.path.isdir(directory):
            rmtree(directory)
        os.makedirs(directory)

    frame_extraction(a,b,trim=True,max_frame=MAX_NUMBER_OF_FRAMES)       # extracts frames to folder b
    for f in os.listdir(b):
        if f.endswith('.avi'):
            directory = os.path.join(c,f) + os.sep
            if os.path.isdir(directory):
                rmtree(directory)
            os.makedirs(directory)
            frames = glob.glob(os.path.join(b,f,'*.png'))
            for fr in frames:
                image_chop(fr, directory, 256)         # Takes subsections of images and saves them to the "ready" folder

    # for i in range(5,30):
    #     diff_imager(b,c, i, i+5,to_chop=True,chopsize=299)        # generates difference images from 5 to 30 with little delta = 5

#    randomize_names(c)      # randomize names for test/val purposes



#From the base directory find all the folders with avi images in them.
#Script to analyze the
base_dir = '/media/hd1/unet_model_search_05082020/unet-master/data_gfp/'
a_fils = glob.glob(base_dir + 'All*')
# a_fils = glob.glob(base_dir + 'BF/Susceptible vs resistant/Resistant/Round 1')
b_fils = [i + os.sep + 'extracted_frames' + os.sep for i in a_fils]
c_fils = [i + os.sep + 'ready_data' + os.sep for i in a_fils]

pool = multiprocessing.Pool(processes=2)
inputs = zip(a_fils,b_fils,c_fils)
outputs = pool.starmap(image_preprocessing, inputs)

    
    


# fils = glob.glob('All*/*/*/*.png')
# for f in fils:
#     directory = f.split('/')[0] + os.sep + 'ready_data' + os.sep + f.split('/')[2] + os.sep
#     if not os.path.isdir(directory):
#         os.makedirs(directory)

#     image_chop(f, directory, 256)         # Takes subsections of images and saves them to the "ready" folder
