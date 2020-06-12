#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 10:58:42 2020

@author: meyerct6
"""


#Import packages for image processing
from mlAbRes.image_preprocessing import frame_extraction,diff_imager_hyper
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
from shutil import rmtree
import multiprocessing


# =============================================================================
# Preprocess data
# =============================================================================
import sys

MAX_NUMBER_OF_FRAMES = int(sys.argv[2])
MAX_LITTLE_DELTA = int(sys.argv[1])
base_dir = sys.argv[3]
#Processing function
def image_preprocessing(a,b,c):
    for directory in [b,c]:
        if os.path.isdir(directory):
            rmtree(directory)
        os.makedirs(directory)
    frame_extraction(a,b,trim=True,max_frame=MAX_NUMBER_OF_FRAMES)       # extracts frames to folder b
    diff_imager_hyper(b, c, MAX_NUMBER_OF_FRAMES, MAX_LITTLE_DELTA,chopsize=256)

#From the base directory find all the folders with avi images in them.
#Script to analyze the
#base_dir = '/home/meyerct6/Data/ml_antibiotics/iphone_bacteria/kralj_videos/*/Hetero resistance/*/'
a_fils = glob.glob(base_dir)
# a_fils = glob.glob(base_dir + 'BF/Susceptible vs resistant/Resistant/Round 1')
b_fils = [i + os.sep + 'extracted_frames' + os.sep for i in a_fils]
c_fils = [i + os.sep + 'ready_data' + os.sep for i in a_fils]

num_processes = 24 if len(a_fils)>24 else len(a_fils)
pool = multiprocessing.Pool(processes=num_processes)
inputs = zip(a_fils,b_fils,c_fils)
outputs = pool.starmap(image_preprocessing, inputs)




