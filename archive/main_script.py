#Trains a new model from scratch.
#Libs
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import keras as kr
import tensorflow_hub as hub

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy
import matplotlib.pyplot as plt

##########################
## File Import
train_dir = "/Library/ML Data/Antibiotic videos/Train"
val_dir = "/Library/ML Data/Antibiotic videos/Test"

train_treated_dir = "/Library/ML Data/Antibiotic videos/Train/Treated Selected Frames train"
train_untreated_dir = "/Library/ML Data/Antibiotic videos/Train/Untreated Selected Frames train"

val_treated_dir = "/Library/ML Data/Antibiotic videos/Test/Treated Selected Frames val"
val_untreated_dir = "/Library/ML Data/Antibiotic videos/Test/Untreated Selected Frames val"

num_treated_tr = len(os.listdir(train_treated_dir))
num_untreated_tr = len(os.listdir(train_untreated_dir))
num_treated_vl = len(os.listdir(train_treated_dir))
num_untreated_vl = len(os.listdir(train_untreated_dir))

total_train = num_treated_tr + num_untreated_tr
total_val = num_treated_vl + num_untreated_vl

##########################

# Variables

batch_size = 4
epochs = 20 #50?
IMG_HEIGHT = 150
IMG_WIDTH = 150

##########################

#Data preparation (used to generate usable data from images)
#image_gen_train = ImageDataGenerator(
#    rescale=1. / 255,
#    rotation_range=45,
#    width_shift_range=.15,
#    height_shift_range=.15,
#    horizontal_flip=True,
#    zoom_range=0.5
#)

image_gen_train = ImageDataGenerator(rescale=1. / 255)

validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

#Define the model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

#Compile using ADAM
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Displays a summary of the model in console.
model.summary()

#Model Training
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

#Plot results with matplotlib
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
