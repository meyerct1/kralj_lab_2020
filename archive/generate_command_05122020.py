#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:19:13 2020

@author: meyerct6
"""



ld = [1,5,60]
tt = [15,30,120]
with open('commands.sh','w') as f:
	f.write(r'#!/bin/bash')
	f.write('\n')
cnt = 1
for l,t in zip(ld,tt):
	with open('commands.sh','a') as f:
	        f.write('python timesweep_analysis_preprocess.py ' + str(l) + ' ' + str(t) +'\n')
	        f.write('python timesweep_analysis_unet.py ' + str(l) + ' ' + str(t) + '\n')
	        f.write("echo 'Finished " + str(cnt) + "'\n")
	        cnt = cnt + 1

