#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:50:21 2020

@author: meyerct6
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
font = {'family' : 'arial',
        'weight':'normal',
        'size'   : 14}
axes = {'linewidth': 2}
rc('font', **font)
rc('axes',**axes)


df = pd.read_csv('05282020_all_results_little-delta_nested_total-time_15.csv')


# df['t1'] = [int(i.split('diff(')[1].split(')')[0].split(' - ')[0]) for i in df['im1_name']]
# df['t2'] = [int(i.split('diff(')[1].split(')')[0].split(' - ')[1]) for i in df['im1_name']]
df['conc'] = [int(i.split('Kan(')[1].split(')')[0]) for i in df['im1_name']]
df['label'] = [i.split('/')[-4] for i in df['im1_name']]
df.loc[df['conc']==0,'conc'] = df.loc[df['conc']!=0,'conc'].min()/10.
df['conc'] = np.log10(df['conc'])
df['round'] =[int(i.split('/')[-5][-1]) for i in df['im1_name']]

for r in df['round'].unique():


    t_thresh = 10
    plt.figure(figsize = (15,6))
    col = ['r','b']
    for e1,l in enumerate(df['label'].unique()):
        ax = plt.subplot(1,len(df['label'].unique()),e1+1)
        df_sub = df[(df['label']==l)&(df['round']==r)]
        bx = df_sub.boxplot(column='per_res_gfp',by='conc',
                            ax=ax,showfliers=False,showcaps=False,
                            fontsize=14,return_type='dict')
        for element in ['boxes','fliers','medians','whiskers']:
            for key in bx.keys():
                for item in bx[key][element]:
                    item.set_linewidth(4)
    
        for element in ['boxes','fliers','whiskers']:
            for key in bx.keys():
                for item in bx[key][element]:
                    item.set_color(col[e1])
    
        for element in ['medians']:
            for key in bx.keys():
                for item in bx[key][element]:
                    item.set_color('k')
    
        # plt.xlim((np.log10(df['conc'].min()),np.log10(df['conc'].max())))
        plt.ylim((0,1.1))
        plt.xlabel('log(Kan)[uM]')
        plt.ylabel('Percent Resistant according to ThT\nModel' +r'($\delta$=1,T=15)')
        plt.title(l + ' Round ' + str(r))
        ax.set_xticklabels([np.round(float(ax.get_xticklabels()[i].get_text()),2) for i in range(len(ax.get_xticks()))])
    
    
    
    plt.tight_layout()





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
font = {'family' : 'arial',
        'weight':'normal',
        'size'   : 12}
axes = {'linewidth': 2}
rc('font', **font)
rc('axes',**axes)


df = pd.read_csv('05222020_1_results_little-delta_1_total-time_15.csv')
# df = df[df['per_sig_back_gfp']<.3]

df['t1'] = [int(i.split('diff(')[1].split(')')[0].split(' - ')[0]) for i in df['im1_name']]
df['t2'] = [int(i.split('diff(')[1].split(')')[0].split(' - ')[1]) for i in df['im1_name']]
df['conc'] = [int(i.split('Kan(')[1].split(')')[0]) for i in df['im1_name']]
df['label'] = [i.split('/')[-4] for i in df['im1_name']]
df.loc[df['conc']==0,'conc'] = df.loc[df['conc']!=0,'conc'].min()/10.
df['conc'] = np.log10(df['conc'])
df['fov'] = [i.split('ready_data/')[1].split('Kan(')[0][0:-1] for i in df['im1_name']]
df['conc'] = np.round(df['conc'],2)

t_thresh = 10
col = ['r','b']
for e1,l in enumerate(df['label'].unique()):
    plt.figure(figsize = (15,6))
    ax = plt.subplot(111)
    df_sub = df[(df['label']==l)&(df['t2']>t_thresh)]
    bx = df_sub.boxplot(column='per_res_gfp',by=['conc','fov'],
                                        ax=ax,showfliers=False,showcaps=False,
                                        fontsize=12,return_type='dict')
    for element in ['boxes','fliers','medians','whiskers']:
        for key in bx.keys():
            for item in bx[key][element]:
                item.set_linewidth(4)

    for element in ['boxes','fliers','whiskers']:
        for key in bx.keys():
            for item in bx[key][element]:
                item.set_color(col[e1])

    for element in ['medians']:
        for key in bx.keys():
            for item in bx[key][element]:
                item.set_color('k')

    # plt.xlim((np.log10(df['conc'].min()),np.log10(df['conc'].max())))
    plt.ylim((0,1.1))
    plt.xlabel('log(Kan)[uM]')
    plt.ylabel('Percent Resistant according to ThT\nModel' +r'($\delta$=5,T=30)')
    plt.title(l)
    plt.tight_layout()