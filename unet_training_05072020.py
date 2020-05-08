#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:08:24 2020

@author: meyerct6
"""

#Import the U-Net model and data functions
from model import *
from data import *
import glob
import os

#The RTX2070 + CUDA10 causes a problem which is solved by this configuration:
#See https://kobkrit.com/using-allow-growth-memory-option-in-tensorflow-and-keras-dc8c8081bc96
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


#Set the batch size
batch_size = 8
#Path to the training data
train_path ='/home/meyerct6/Data/ml_antibiotics/unet-master/data_gfp/unet_data/train_little-delta_5_total-time_30'
#train_path ='/home/meyerct6/Data/ml_antibiotics/unet-master/data/membrane/train'
#What folders have the image and the label
image_folder = 'image'
mask_folder = 'label'
#The number of steps per epoch is the number of training images divided by the batch size
steps_per_epoch = int(len(glob.glob(train_path+os.sep+image_folder+os.sep+'*.png'))/batch_size)
#How many classes are there?  3 for background, resistant, sensitive
num_class = 3

#Augment the data with some defined alterations
#These are defined by U-Net package
aug_dict = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#Create a data generator
myData = trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = num_class,save_to_dir = None,target_size = (256,256),seed = 1)

#Instantiate the model
model = unet()
#Save the model after each epoch
model_checkpoint = ModelCheckpoint('/home/meyerct6/Data/ml_antibiotics/unet-master/data_gfp/unet_little-delta_5_total-time_30.hdf5', monitor='loss',verbose=1, save_best_only=True)
#Fit the model with 5 epochs
H = model.fit_generator(myData,steps_per_epoch=steps_per_epoch,epochs=5,callbacks=[model_checkpoint])


test_path ='/home/meyerct6/Data/ml_antibiotics/unet-master/data_gfp/unet_data/test_little-delta_5_total-time_30'

myDataTest = trainGenerator(batch_size,test_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = num_class,save_to_dir = None,target_size = (256,256),seed = 1)

steps = int(len(glob.glob(test_path+os.sep+image_folder+os.sep+'*.png'))/batch_size)
model.evaluate_generator(myDataTest,steps,verbose=1)


#Test predictions
test_path ='/home/meyerct6/Data/ml_antibiotics/unet-master/data_gfp/unet_data/test_little-delta_5_total-time_30/image'
number_of_samples = 30
testGene = testGenerator(test_path,start_im=50,end_im=80)
model = unet()
model.load_weights("/home/meyerct6/Data/ml_antibiotics/unet-master/data_gfp/unet_little-delta_5_total-time_30.hdf5")
results = model.predict_generator(testGene,number_of_samples,verbose=1)
saveResult('/home/meyerct6/Data/ml_antibiotics/unet-master/data/unet_data/test_little-delta_5_total-time_30/predict',results,start_im=50,flag_multi_class = True,num_class = 3)

