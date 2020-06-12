#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:19:13 2020

@author: meyerct6
"""


ld = [1,5,5,30];
tt = [15,15,30,60];

with open('commands.sh','w') as f:
	f.write(r'#!/bin/bash')
	f.write('\n')
cnt = 1
for ch in ['gfp','bf']:
	for l,t in zip(ld,tt):
		with open('commands.sh','a') as f:
			f.write('python unet_training_06052020_timesweep.py ' + str(l) + ' ' + str(t) + ' ' +  ch + '\n')
			f.write("echo 'Finished " + str(cnt) + "'\n")
			cnt = cnt + 1

