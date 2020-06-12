#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:36:07 2020

@author: meyerct6
"""
#Import the U-Net model and data functions
from model import unet
from data import predictGenerator
import pandas as pd
import numpy as np
import glob

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

MAX_NUMBER_OF_FRAMES = int(sys.argv[2])
LITTLE_DELTA = int(sys.argv[1])
base_dir1 = sys.argv[3]
base_dir2 = sys.argv[4]
fils1 = np.sort(glob.glob(base_dir1))
fils2 = np.sort(glob.glob(base_dir2))
fil_cnter = range(len(fils1))

# #Check to make sure files line up
# for f1,f2,c in zip(fils1,fils2,fil_cnter):
#     t1 = '/'.join(f1.split('/')[-4:-2]);    t2 = '/'.join(f2.split('/')[-4:-2])
#     v1 = '-'.join(f1.split('/')[-1].split('-')[1:]);    v2 = '-'.join(f2.split('/')[-1].split('-')[1:])
#     w1 = '_'.join(f1.split('/')[-1].split('_')[0:2]);    w2 = '_'.join(f1.split('/')[-1].split('_')[0:2])
#     tmp1 = '_'.join([t1,v1,w1]);    tmp2 = '_'.join([t2,v2,w2])
#     if tmp1!=tmp2:
#         print('error on file index' + str(c))
#         break

#Instantiate the model
model1 = unet()
model2 = unet()
#Load the file weights
ch='gfp'
weights1_file = '/media/hd1/unet_model_search_05082020/unet-master/data_'+ch+'/unet_little-delta_'+str(LITTLE_DELTA)+'_total-time_'+str(MAX_NUMBER_OF_FRAMES)+'.hdf5'
ch = 'bf'
weights2_file = '/media/hd1/unet_model_search_05082020/unet-master/data_'+ch+'/unet_little-delta_'+str(LITTLE_DELTA)+'_total-time_'+str(MAX_NUMBER_OF_FRAMES)+'.hdf5'
model1.load_weights(weights1_file)
model2.load_weights(weights2_file)

#Build the output dataframe
W = .5; #Weight between the models
df = pd.DataFrame(np.zeros((len(fils1),14)))
df.columns = ['per_sig_back_gfp','per_sig_back_bf','per_sig_back_combo','per_res_bf','per_res_gfp','per_back_bf','per_back_gfp','im1_name','im2_name','per_sus_bf','per_sus_gfp','per_sus_combo','per_res_combo','per_back_combo']


#Batch size of ~30 is the maximal size
batch_size = 30
print('Total Number of Images: ' + str(len(fils2)))
for f1,f2,cnt in zip(chunked_iterable(fils1,size=batch_size),chunked_iterable(fils2,size=batch_size),chunked_iterable(fil_cnter,size=batch_size)):
    print('Currently working on images: ' + str(cnt[0]) + '-' + str(cnt[-1]))

    df_file = pd.DataFrame({'file_gfp':f1,'file_bf':f2})
    #Use a generator which uses flow_from_dataframe (df_file_sub)
    predict_generator1 = predictGenerator(len(cnt),df_file,x_col='file_gfp',image_color_mode = "grayscale",target_size = (256,256))
    predict_generator2 = predictGenerator(len(cnt),df_file,x_col='file_bf',image_color_mode = "grayscale",target_size = (256,256))

    predict_generator1.reset()
    predict_generator2.reset()
    results1 = model1.predict_generator(predict_generator1,steps=1)
    results2 = model2.predict_generator(predict_generator2,steps=1)
    results3 = W*results1 + (1-W)*results2

    r1 = np.argmax(results1,3)
    r2 = np.argmax(results2,3)
    r3 = np.argmax(results3,3)
    #Begin adding information to the dataframe

    df.loc[cnt,'per_back_gfp']      = np.sum(np.sum(r1==0,2),1)/256**2
    df.loc[cnt,'per_back_bf']       = np.sum(np.sum(r2==0,2),1)/256**2
    df.loc[cnt,'per_back_combo']    = np.sum(np.sum(r3==0,2),1)/256**2

    df.loc[cnt,'per_res_gfp']       = np.sum(np.sum(r1==2,2),1)/np.sum(np.sum(r1!=0,2),1)
    df.loc[cnt,'per_res_bf']        = np.sum(np.sum(r2==2,2),1)/np.sum(np.sum(r2!=0,2),1)
    df.loc[cnt,'per_res_combo']     = np.sum(np.sum(r3==2,2),1)/np.sum(np.sum(r3!=0,2),1)

    df.loc[cnt,'per_sig_back_gfp']     = np.sum(np.sum(r1!=0,2),1)/np.sum(np.sum(r1==0,2),1)
    df.loc[cnt,'per_sig_back_bf']      = np.sum(np.sum(r2!=0,2),1)/np.sum(np.sum(r2==0,2),1)
    df.loc[cnt,'per_sig_back_combo']   = np.sum(np.sum(r3!=0,2),1)/np.sum(np.sum(r3==0,2),1)

    df.loc[cnt,'im1_name'] = f1
    df.loc[cnt,'im2_name'] = f2

    # df.loc[cnt,'t1'] = [int(i.split('(')[1].split(')')[0].split(' - ')[0]) for i in f1]
    # df.loc[cnt,'t2'] = [int(i.split('(')[1].split(')')[0].split(' - ')[1]) for i in f1]

    # df.loc[cnt,'label'] = [i.split('/')[-4] for i in f1]



df.to_csv('05222020_1_results_little-delta_'+str(LITTLE_DELTA)+'_total-time_'+str(MAX_NUMBER_OF_FRAMES)+'.csv')
