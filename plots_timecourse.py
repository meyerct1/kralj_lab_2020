#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 06:22:42 2020

@author: meyerct6
"""


import pandas as pd
import matplotlib.pyplot as plt
ld = [1,5,60]
tt = [15,30,120]
df = []
for l,t in zip(ld,tt):
    df.append(pd.read_csv('results_little-delta_'+str(l)+'_total-time_'+str(t)+'.csv'))
    df[-1].groupby(['label','t2'])['per_res_bf'].median().unstack(0).plot()
    plt.ylabel('Percent Resistance According to BF model')
    plt.xlabel('Time Point of 2nd image in difference (min)')
    plt.title('Model: Little-Delta ' + str(l) + ' Total-time ' + str(t))




import pandas as pd
import matplotlib.pyplot as plt
ld = [1]
tt = [15]
df = []
for l,t in zip(ld,tt):
    df.append(pd.read_csv('05082020_results_little-delta_'+str(l)+'_total-time_'+str(t)+'.csv'))
    df[-1].groupby(['label','t2'])['per_res_gfp'].median().unstack(0).plot()
    plt.ylabel('Percent Resistance According to GFP model')
    plt.xlabel('Time Point of 2nd image in difference (min)')
    plt.title('Model: Little-Delta ' + str(l) + ' Total-time ' + str(t))