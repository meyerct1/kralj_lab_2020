#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:52:50 2020

@author: meyerct6
"""

#Import packages for image processing
from mlAbRes.image_preprocessing import frame_extraction,diff_imager
import os
import glob
from shutil import rmtree
import multiprocessing

#Import the U-Net model and data functions
from model import unet
from data import predictGenerator
import pandas as pd
import numpy as np

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
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
# =============================================================================
# Preprocess data
# =============================================================================

MAX_NUMBER_OF_FRAMES = 30
LITTLE_DELTA = 5

# =============================================================================
# Prediction....
# =============================================================================
fils1 = np.sort(glob.glob('/home/meyerct6/Data/ml_antibiotics/iphone_bacteria/kralj_videos/GFP/unet_data/test_little-delta_5_total-time_30/image/*.png'))
fils2 = np.sort(glob.glob('/home/meyerct6/Data/ml_antibiotics/iphone_bacteria/kralj_videos/BF/unet_data/test_little-delta_5_total-time_30/image/*.png'))
fil_cnter = range(len(fils1))


#Instantiate the model
model1 = unet()
#Load the file weights
ch='gfp'
weights1_file = '/media/hd1/unet_model_search_05082020/unet-master/data_'+ch+'/unet_little-delta_'+str(LITTLE_DELTA)+'_total-time_'+str(MAX_NUMBER_OF_FRAMES)+'.hdf5'
model1.load_weights(weights1_file)

#Build the output dataframe
W = .5; #Weight between the models
df = pd.DataFrame(np.zeros((len(fils1),14)))
df.columns = ['label','per_res_bf','per_res_gfp','per_back_bf','per_back_gfp','t1','t2','im1_name','im2_name','per_sus_bf','per_sus_gfp','per_sus_combo','per_res_combo','per_back_combo']
df_file = pd.DataFrame({'file_gfp':fils1})


#Batch size of ~1000 is the maximal size
batch_size = 1000
print('Total Number of Images: ' + str(len(fils2)))
for f1,f2,cnt in zip(chunked_iterable(fils1,size=batch_size),chunked_iterable(fils2,size=batch_size),chunked_iterable(fil_cnter,size=batch_size)):
    print('Currently working on images: ' + str(cnt[0]) + '-' + str(cnt[-1]))

    df_file_sub = df_file.loc[list(cnt)].reset_index()
    #Use a generator which uses flow_from_dataframe (df_file_sub)
    predict_generator1 = predictGenerator(1,df_file_sub,x_col='file_gfp',image_color_mode = "grayscale",target_size = (256,256))

    results1 = model1.predict_generator(predict_generator1,steps=len(cnt))

    r1 = np.argmax(results1,3)

    #Begin adding information to the dataframe

    df.loc[cnt,'per_back_gfp']      = np.sum(np.sum(r1==0,2),1)/256**2


    df.loc[cnt,'per_res_gfp']       = np.sum(np.sum(r1==2,2),1)/np.sum(np.sum(r1!=0,2),1)

    df.loc[cnt,'im1_name'] = f1




# #Batch size of ~1000 is the maximal size
# batch_size = 1000
# print('Total Number of Images: ' + str(len(fils2)))
# for f1,f2,cnt in zip(chunked_iterable(fils1,size=batch_size),chunked_iterable(fils2,size=batch_size),chunked_iterable(fil_cnter,size=batch_size)):
#     print('Currently working on images: ' + str(cnt[0]) + '-' + str(cnt[-1]))

#     df_file_sub = df_file.loc[list(cnt)].reset_index()
#     #Use a generator which uses flow_from_dataframe (df_file_sub)
#     predict_generator1 = predictGenerator(1,df_file_sub,x_col='file_gfp',image_color_mode = "grayscale",target_size = (256,256))
#     predict_generator2 = predictGenerator(1,df_file_sub,x_col='file_bf',image_color_mode = "grayscale",target_size = (256,256))

#     results1 = model1.predict_generator(predict_generator1,steps=len(cnt))
#     results2 = model2.predict_generator(predict_generator2,steps=len(cnt))
#     results3 = W*results1 + (1-W)*results2

#     r1 = np.argmax(results1,3)
#     r2 = np.argmax(results2,3)
#     r3 = np.argmax(results3,3)
#     #Begin adding information to the dataframe
#     for e1,c in enumerate(cnt):
#         df.loc[c,'per_back_bf']   =  np.sum(r2[e1,:,:]==0)/np.sum(r2[e1,:,:]!=-1)
#         df.loc[c,'per_back_gfp']  =  np.sum(r1[e1,:,:]==0)/np.sum(r1[e1,:,:]!=-1)
#         df.loc[c,'per_res_bf']    =  np.sum(r2[e1,:,:]==2)/np.sum(r2[e1,:,:]!=0)
#         df.loc[c,'per_res_gfp']   =  np.sum(r1[e1,:,:]==2)/np.sum(r1[e1,:,:]!=0)
#         df.loc[c,'per_sus_bf']    =  np.sum(r2[e1,:,:]==1)/np.sum(r2[e1,:,:]!=0)
#         df.loc[c,'per_sus_gfp']   =  np.sum(r1[e1,:,:]==1)/np.sum(r1[e1,:,:]!=0)
#         df.loc[c,'per_back_combo'] =  np.sum(r3[e1,:,:]==0)/np.sum(r3[e1,:,:]!=-1)
#         df.loc[c,'per_res_combo']  =  np.sum(r3[e1,:,:]==2)/np.sum(r3[e1,:,:]!=0)
#         df.loc[c,'per_sus_combo']  =  np.sum(r3[e1,:,:]==1)/np.sum(r3[e1,:,:]!=0)
    
    
#         df.loc[c,'im1_name'] = f1[e1]
#         df.loc[c,'im2_name'] = f2[e1]

#         tmp = f1[e1].split('(')[1].split(')')[0]
    
#         df.loc[c,'t1'] = int(tmp.split(' - ')[0])
#         df.loc[c,'t2'] = int(tmp.split(' - ')[1])
    
#         tmp = f1[e1].split('/')[-4]
#         if tmp == 'All susceptible':
#             df.loc[c,'label'] = 0
#         elif tmp == 'All resistant':
#             df.loc[c,'label'] = 1
#         elif tmp == '1 to 0.1 sus to res':
#             df.loc[c,'label'] = .1
#         elif tmp == '1 to 0.01 sus to res':
#             df.loc[c,'label']= .01
#         elif tmp == '1 to 1 sus to res':
#             df.loc[c,'label'] = .5
#         else:
#             raise('Error')

