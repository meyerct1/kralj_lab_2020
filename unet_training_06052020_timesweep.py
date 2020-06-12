#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:08:24 2020

@author: meyerct6
"""

#Import the U-Net model and data functions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from AST_unet.unet_model import unet,ModelCheckpoint
from AST_unet.tensor_data import trainGenerator
import glob
import sys
import pandas as pd
import time
import numpy as np

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

#Get input parameters
little_delta = sys.argv[1]
total_time = sys.argv[2]
channel = sys.argv[3]


t1 = time.time()

#Set the batch size
batch_size = 8
#Path to the training data
train_path ='/media/hd1/unet_model_training_data/unet-master/data_'+channel+'/unet_data/train_little-delta_'+little_delta+'_total-time_'+total_time
#train_path ='/home/meyerct6/Data/ml_antibiotics/unet-master/data/membrane/train'
#What folders have the image and the label
image_folder = 'image'
mask_folder = 'label'
#The number of steps per epoch is the number of training images divided by the batch size
steps_per_epoch = 1000 #int(len(glob.glob(train_path+os.sep+image_folder+os.sep+'*.png'))/batch_size)
#How many classes are there?  3 for background, resistant, sensitive
num_class = 3

#Augment the data with some defined alterations
#These are defined by U-Net package
#The mask must have the same augmentation applied to it excepting brightness
aug_dict_image = dict(shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest',
                    brightness_range = (0,.1))
aug_dict_mask = dict(shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#Create a data generator
myData = trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict_image,aug_dict_mask,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = num_class,save_to_dir = None,target_size = (256,256),seed = 1)

#Instantiate the model
model = unet()
#Save the model after each epoch
model_checkpoint = ModelCheckpoint('/media/hd1/unet_model_training_data/unet-master/data_'+channel+'/06082020_unet_little-delta_'+little_delta+'_total-time_'+total_time+'.hdf5', monitor='loss',verbose=1, save_best_only=True)
#Fit the model with 5 epochs
try:
    H1 = model.fit_generator(myData,steps_per_epoch=steps_per_epoch,epochs=2,callbacks=[model_checkpoint])

	#Test the model
    test_path ='/media/hd1/unet_model_training_data/unet-master/data_'+channel+'/unet_data/test_little-delta_'+little_delta+'_total-time_'+total_time
    myDataTest = trainGenerator(batch_size,test_path,image_folder,mask_folder,aug_dict_image,aug_dict_mask,image_color_mode = "grayscale",
            		            mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
            		            flag_multi_class = False,num_class = num_class,save_to_dir = None,target_size = (256,256),seed = 1)
    #Test on a 100*batch_size images
    steps = 125 #int(len(glob.glob(test_path+os.sep+image_folder+os.sep+'*.png')))
    H2 = model.evaluate_generator(myDataTest,steps,verbose=1)
    status = 'complete'
    train_accuracy = H1.history['accuracy'][-1]
    train_loss = H1.history['loss'][-1]
    test_accuracy = H2[1]
    test_loss = H2[0]
except:
    fil = '/media/hd1/unet_model_training_data/unet-master/data_'+channel+'/unet_little-delta_'+little_delta+'_total-time_'+total_time+'.hdf5'
    if os.path.isfile(fil):
        status = 'early_terminiation'
    else:
        status = 'failure'
    train_accuracy = np.nan
    train_loss = np.nan
    test_accuracy = np.nan
    test_loss = np.nan


t2 = time.time()
T = {'little-delta':int(little_delta),'total_time':int(total_time),'channel':channel,'time':(t2-t1)/60.,'status':status,
     'train_loss':train_loss,'train_accuracy':train_accuracy,
     'test_loss':test_loss,'test_accuracy':test_accuracy}
#Write out results to file...
df = pd.DataFrame([T])

#Append the results to a MasterResults file
if not os.path.isfile('/media/hd1/unet_model_training_data/unet-master/MasterResults.csv'):
    df.to_csv('/media/hd1/unet_model_training_data/unet-master/MasterResults.csv')
else:
    with open('/media/hd1/unet_model_training_data/unet-master/MasterResults.csv', 'a') as f:
        df.to_csv(f, header=False)
