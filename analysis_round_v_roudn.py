#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:05:48 2020

@author: meyerct6
"""

import numpy as np
import seaborn as sns
import pandas as pd 
from matplotlib import cm
import matplotlib.pyplot as plt

import pickle
with open('RvR_results.txt','r') as f:
    res = f.read()
res = np.array([float(i) for i in res.split('\n')[:-1]]).reshape((11,11))

cg = sns.clustermap(pd.DataFrame(res),row_cluster=True,cmap=cm.seismic,cbar_kws={'label':'Accuracy'})
ax = cg.ax_heatmap
ax.set_ylabel('Predict using Model from Round xxx')
ax.set_xlabel('Round predicted')
plt.tight_layout()