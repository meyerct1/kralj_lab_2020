#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:00:23 2020

@author: meyerct6
"""


#Import packages for image processing
from mlAbRes.image_preprocessing import frame_extraction,diff_imager,randomize_names,image_chop
import os
import glob
from shutil import rmtree
import multiprocessing

#Import the U-Net model and data functions
from model import *
from data import *
import glob
import os
import sys
import pandas as pd
import numpy as np
import time

#The RTX2070 + CUDA10 causes a problem which is solved by this configuration:
#See https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

ld = [1,1,1,5,5,5,10,10,10,15,15,15,30,30,30,60,60,60]
tt = [10,15,30,15,30,60,30,60,90,60,90,120,90,120,180,120,180,240]
channel = ['gfp','bf']

ch_l = []
ld_l = []
tt_l = []
st_l = []

for l,t in zip(ld,tt):
    for ch in channel:
        test_path = '/media/hd1/unet_model_search_05082020/unet-master/data_'+ch+'/unet_data/test_little-delta_'+str(l)+'_total-time_'+str(t)+'/image'
        number_of_samples = 5
        testGene = testGenerator(test_path,start_im=50,end_im=80)
        model = unet()
        weights_file = '/media/hd1/unet_model_search_05082020/unet-master/data_'+ch+'/unet_little-delta_'+str(l)+'_total-time_'+str(t)+'.hdf5'
        if os.path.exists(weights_file):
            model.load_weights(weights_file)
            results = model.predict_generator(testGene,number_of_samples,verbose=0)
            if np.max(np.argmax(results,3))==0:
                status = 'fail'
            else:
                status = 'succeed'
        else:
            status = 'fail'

        print(status + ' Channel: ' + ch + ', LD: ' + str(l) + ', TT:' + str(t))
        ch_l.append(ch)
        ld_l.append(l)
        tt_l.append(t)
        st_l.append(status)


x = pd.DataFrame([ch_l,ld_l,tt_l,st_l]).T