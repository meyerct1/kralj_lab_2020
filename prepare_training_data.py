#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:23:19 2020

@author: meyerct6
"""


#Import packages for image processing
from AST_unet.image_preprocessing import frame_extraction,diff_imager,randomize_names,image_chop
import os
import glob
from shutil import rmtree, copyfile
import multiprocessing

MAX_NUMBER_OF_FRAMES = 60
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
    #Copy the segmentation settings file
    copyfile('/media/hd1/unet_model_search_05082020/unet-master/data_gfp/All resistant/ready_data/Settings.csv',a+os.sep+'ready_data'+os.sep + 'Settings.csv')

#From the base directory find all the folders with avi images in them.
#Script to analyze the
base_dir = '/media/hd1/unet_model_training_data/unet-master/data_*/All*/Round*'
a_fils = glob.glob(base_dir)
b_fils = [i + os.sep + 'extracted_frames' + os.sep for i in a_fils]
c_fils = [i + os.sep + 'ready_data' + os.sep for i in a_fils]

pool = multiprocessing.Pool(processes=24)
inputs = zip(a_fils,b_fils,c_fils)
outputs = pool.starmap(image_preprocessing, inputs)

