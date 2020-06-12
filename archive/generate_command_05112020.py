#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:19:13 2020

@author: meyerct6
"""



ld = [10,30,30]
tt = [90,120,180]
ch = ['gfp','bf','bf']
with open('commands.sh','w') as f:
	f.write(r'#!/bin/bash')
	f.write('\n')
cnt = 1
for l,t,c in zip(ld,tt,ch):
	with open('commands.sh','a') as f:
	        f.write('python unet_training_05082020_timesweep.py ' + str(l) + ' ' + str(t) + ' ' +  c + '\n')
	        f.write("echo 'Finished " + str(cnt) + "'\n")
	        cnt = cnt + 1

