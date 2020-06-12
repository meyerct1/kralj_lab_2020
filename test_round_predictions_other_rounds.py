#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:36:07 2020

@author: meyerct6
"""
#Import the U-Net model and data functions
from AST_unet.unet_model import unet
from AST_unet.tensor_data import predictGenerator,trainGenerator
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
tf.random.set_random_seed(1234)
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

total_time = 15
little_delta = 1
ch = 'gfp'
input_cnt = int(sys.argv[1])
#Set the batch size
batch_size = 8
#Path to the training data
#What folders have the image and the label
image_folder = 'image'
mask_folder = 'label'
#The number of steps per epoch is the number of training images divided by the batch size
#How many classes are there?  3 for background, resistant, sensitive
num_class = 3

#Augment the data with some defined alterations
#These are defined by U-Net package
#The mask must have the same augmentation applied to it excepting brightness
aug_dict_image = dict()
aug_dict_mask = dict()

cnt = 0
for i1 in range(11):
    for i2 in range(11):
        if cnt==input_cnt:
            weights_file = '/media/hd1/unet_model_training_data/unet-master/data_'+ch+'/round_'+str(i1)+'_06112020_unet_little-delta_'+str(little_delta)+'_total-time_'+str(total_time)+'.hdf5'
            model = unet(pretrained_weights=weights_file)
            test_path ='/media/hd1/unet_model_training_data/unet-master/data_'+ch+'/unet_data/round_'+str(i2)+'_test_little-delta_'+str(little_delta)+'_total-time_'+str(total_time)
    
        	#Test the model
            myDataTest = trainGenerator(batch_size,test_path,image_folder,mask_folder,aug_dict_image,aug_dict_mask,image_color_mode = "grayscale",
                    		            mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    		            flag_multi_class = False,num_class = num_class,save_to_dir = None,target_size = (256,256),seed = 1)
            #Test on a 100*batch_size images
            steps = 20 #int(len(glob.glob(test_path+os.sep+image_folder+os.sep+'*.png')))
            H2 = model.evaluate_generator(myDataTest,steps,verbose=1)
            with open('RvR_results.txt','a') as f:
                f.write(str(H2[1])+'\n')
            print(str(cnt) + ' ' + str(H2[1]))
        cnt = cnt + 1

# import seaborn as sns
# from matplotlib import cm
# import matplotlib.pyplot as plt
# f = plt.figure()
# ax = plt.subplot(111)

# import pickle

# p1 = pickle.load(open('tmp.p','rb'))
# p2 = pickle.load(open('tmp-1.p','rb'))
# p3 = pickle.load(open('tmp-2.p','rb'))
# tmp = p1+p2+p3
# tmp[tmp>1] = p2[(p1!=0)&(p2!=0)]


# sns.clustermap(pd.DataFrame(tmp),row_cluster=False,cmap=cm.seismic)
# ax.set_ylabel('Predict using Model from Round xxx')
# ax.set_xlabel('Round predicted')
