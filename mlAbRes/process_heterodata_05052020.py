#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:57:42 2020

@author: meyerct6
"""

#Import packages for image processing
from mlAbRes.image_preprocessing import frame_extraction,diff_imager,randomize_names
import os
import glob
from shutil import rmtree
import multiprocessing


#Processing function
def image_preprocessing(a,b,c):
    for directory in [b,c]:
        if os.path.isdir(directory):
            rmtree(directory)
        os.makedirs(directory)

    frame_extraction(a,b,trim=True)       # extracts frames to folder b

    for i in range(5,30):
        diff_imager(b,c, i, i+5,to_chop=True,chopsize=299)        # generates difference images from 5 to 30 with little delta = 5

#    randomize_names(c)      # randomize names for test/val purposes



#From the base directory find all the folders with avi images in them.
#Script to analyze the
base_dir = '/home/meyerct6/Data/ml_antibiotics/iphone_bacteria/kralj_videos/'

a_fils = glob.glob(base_dir + '*/*/*/*')
# a_fils = glob.glob(base_dir + 'BF/Susceptible vs resistant/Resistant/Round 1')
b_fils = [i + os.sep + 'extracted_frames' + os.sep for i in a_fils]
c_fils = [i + os.sep + 'ready_data' + os.sep for i in a_fils]

pool = multiprocessing.Pool(processes=24)
inputs = zip(a_fils,b_fils,c_fils)
outputs = pool.starmap(image_preprocessing, inputs)

    
    
