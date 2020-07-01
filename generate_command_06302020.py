#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:19:13 2020

@author: meyerct6
"""

import numpy as np
ld = [1];
tt = [15];

with open('commands.sh','w') as f:
	f.write(r'#!/bin/bash')
	f.write('\n')
cnt = 1
for ch in ['gfp']:
    for l,t in zip(ld,tt):
        for r in np.arange(1,7):
            with open('commands.sh','a') as f:
                f.write('python unet_training_06302020_roundsweep.py ' + str(l) + ' ' + str(t) + ' ' +  ch + ' ' + str(r) + '\n')
                f.write("echo 'Finished " + str(cnt) + "'\n")
                cnt = cnt + 1

