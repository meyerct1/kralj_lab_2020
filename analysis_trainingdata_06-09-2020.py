#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:58:35 2020

@author: meyerct6
"""

import os

ld = [1,5,5,30];
tt = [15,15,30,60];

with open('commands.sh','w') as f:
	f.write(r'#!/bin/bash')
	f.write('\n')
cnt = 1
for ch in ['bf','gfp']:
    for l,t in zip(ld,tt):
        im_path ='/media/hd1/unet_model_training_data/unet-master/data_'+ch+'/unet_data/test_little-delta_'+str(l)+'_total-time_'+str(t)+os.sep+'image'+os.sep+'*.png'
        with open('commands.sh','a') as f:
            f.write('python fit_data_unet.py ' + str(l) + ' ' + str(t) + ' ' +  ch + ' ' + im_path + '\n')
            f.write("echo 'Finished " + str(cnt) + "'\n")
            cnt = cnt + 1
