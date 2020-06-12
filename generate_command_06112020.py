#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:19:13 2020

@author: meyerct6
"""


#ld = [1];
#tt = [15];
#rnd = range(11)
#with open('commands.sh','w') as f:
#	f.write(r'#!/bin/bash')
#	f.write('\n')
#cnt = 1
#for r in range(11):
#    for ch in ['gfp']:
#    	for l,t in zip(ld,tt):
#    		with open('commands.sh','a') as f:
#    			f.write('python unet_training_06112020_roundsweep.py ' + str(l) + ' ' + str(t) + ' ' +  ch + ' ' + str(r) + '\n')
#    			f.write("echo 'Finished " + str(cnt) + "'\n")
#    			cnt = cnt + 1




with open('commands.sh','w') as f:
	f.write(r'#!/bin/bash')
	f.write('\n')
cnt = 0
for i1 in range(11):
    for i2 in range(11):
        with open('commands.sh','a') as f:
            f.write('python test_round_predictions_other_rounds.py ' + str(cnt) + '\n')
            f.write("echo 'Finished " + str(cnt) + "'\n")
            cnt = cnt + 1

