#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:10:28 2020

@author: meyerct6
"""

#Import the U-Net model and data functions
from AST_unet.unet_model import unet
from AST_unet.tensor_data import predictGenerator
import pandas as pd
import numpy as np
import glob
import os

import itertools
def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

#The RTX2070 + CUDA10 causes a problem which is solved by this configuration:
#See https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# =============================================================================
# Prediction....
# =============================================================================
import sys
import os

ld = [1,5,5,30];
tt = [15,15,30,60];

with open('commands.sh','w') as f:
	f.write(r'#!/bin/bash')
	f.write('\n')
cnt = 1
for ch in ['bf','gfp']:
    for LITTLE_DELTA,MAX_NUMBER_OF_FRAMES in zip(ld,tt):
        base_dir ='/media/hd1/unet_model_training_data/unet-master/data_'+ch+'/unet_data/test_little-delta_'+str(LITTLE_DELTA)+'_total-time_'+str(MAX_NUMBER_OF_FRAMES)+os.sep+'image'+os.sep+'*.png'
        
        fils1 = np.sort(glob.glob(base_dir))[:500]
        fil_cnter = range(len(fils1))[:500]
        
        #Instantiate the model
        weights_file = '/media/hd1/unet_model_training_data/unet-master/data_'+ch+'/06082020_unet_little-delta_'+str(LITTLE_DELTA)+'_total-time_'+str(MAX_NUMBER_OF_FRAMES)+'.hdf5'
        model = unet(pretrained_weights=weights_file)
        
        #Build the output dataframe
        df = pd.DataFrame(np.zeros((len(fils1),6)))
        df.columns = ['per_res','per_back','im_name','channel','little_delta','total_time']
        df['channel']=ch
        df['little_delta']=LITTLE_DELTA
        df['total_time']=MAX_NUMBER_OF_FRAMES
        
        
        #Batch size of ~30 is the maximal size
        batch_size = 8
        print('Total Number of Images: ' + str(len(fils1)))
        for f1,cnt in zip(chunked_iterable(fils1,size=batch_size),chunked_iterable(fil_cnter,size=batch_size)):
            print('Currently working on images: ' + str(cnt[0]) + '-' + str(cnt[-1]))
        
            df_file = pd.DataFrame({'file':f1})
            #Use a generator which uses flow_from_dataframe (df_file_sub)
            predict_generator1 = predictGenerator(len(cnt),df_file,x_col='file',image_color_mode = "grayscale",target_size = (256,256))
        
            predict_generator1.reset()
            results1 = model.predict_generator(predict_generator1,steps=1)
        
            r1 = np.argmax(results1,3)
            #Begin adding information to the dataframe
        
            df.loc[cnt,'per_back']      = np.sum(np.sum(r1==0,2),1)/256**2
            df.loc[cnt,'per_res']       = np.sum(np.sum(r1==2,2),1)/np.sum(np.sum(r1!=0,2),1)
            df.loc[cnt,'im_name']       = f1
        
        
        #Append the results to a MasterResults file
        if not os.path.isfile('unet_prediction.csv'):
            df.to_csv('unet_prediction.csv')
        else:
            with open('unet_prediction.csv', 'a') as f:
                df.to_csv(f, header=False)

