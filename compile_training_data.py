#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:19:02 2020

@author: meyerct6
"""

import pandas as pd
import glob
import numpy as np
import os
from shutil import copyfile
import multiprocessing

df= pd.read_csv('compile_training_data.csv')

#check to make sure files exist
for i in df.index:
    fils = glob.glob(df.loc[i,'Source'])
    if len(fils)==0:
        print(i)

sour = []
dest = []
for i in df.index:
    fils = glob.glob(df.loc[i,'Source'])
    sour = sour + fils
    dest = dest + list(np.repeat(df.loc[i,'Location'],len(fils)))

for d in np.unique(dest):
    if not os.path.exists(d):
        os.makedirs(d)

def parallel_copy(a,b,c):
    print(c)
    if os.path.isfile(b+os.sep+str(c)+'.avi'):
        os.remove(b+os.sep+str(c)+'.avi')
    copyfile(a,b+os.sep+str(c)+'.avi')

for a,b,c in zip(sour,dest,range(len(sour))):
    parallel_copy(a,b,c)

