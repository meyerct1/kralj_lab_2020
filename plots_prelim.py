#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:28:48 2020

@author: meyerct6
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/media/hd1/unet_model_search_05082020/unet-master/MasterResults.csv')
df = df.sort_values('test_accuracy').drop_duplicates(['little-delta','total_time','channel'])


plt.figure()
x = df.groupby('channel')['test_accuracy'].mean()
ax = x.plot.bar()
ax.set_ylabel("Accuracy")
ax.set_xticklabels(['BF','ThT'],rotation=0)
ax.set_ylim([.7,1])
ax.set_title('Average over all runs')


plt.figure()
df_sub = df[df['channel']=='gfp']
plt.scatter(df_sub['little-delta'],df_sub['total_time'],50,c=df_sub['test_accuracy'],edgecolor='k')
plt.colorbar(label='Accuracy')
plt.ylabel('Total Time')
plt.xlabel(r'$\Delta$')
plt.title('Test Accuracy')