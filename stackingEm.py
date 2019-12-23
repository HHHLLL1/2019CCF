# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:56:25 2019

@author: Lenovo
"""

import os
import numpy as np
import pandas as pd
import tqdm
from sklearn.linear_model import LogisticRegression




###############################################################################


#aaa = pd.read_csv('./sub/规则Em0.6929.csv')
#bbb = pd.read_csv('./sub/lgb.csv') 
#ccc = pd.read_csv('./sub/cbt.csv') 
#
#pre = []
##for i in range(120):
##    a = cbt_pre.iloc[i, 1:].values
##    b = lgb_pre.iloc[i, 1:].values
##    c = xgb_pre.iloc[i, 1:].values
##    
##    if (a.max()-a.min()) > (b.max()-b.min()):
##        if (a.max()-a.min()) > (c.max()-c.min()):
##            pre.append(a)
##        else:
##            pre.append(c)
##    else:
##        if (b.max()-b.min()) > (c.max()-c.min()):
##            pre.append(b)
##        else:
##            pre.append(c)
#
#for i in range(120):
#    a = aaa.iloc[i, 1:].values
#    b = bbb.iloc[i, 1:].values
#    c = ccc.iloc[i, 1:].values
#    
#    if a.max() > b.max():
#        if a.max() > c.max():
#            pre.append(a)
#        else:
#            pre.append(c)
#    else:
#        if b.max() > c.max():
#            pre.append(b)
#        else:
#            pre.append(c)
#
#pre = np.array(pre)
#
#sub = aaa[['Group']]
#prob_cols = [i for i in aaa.columns if i not in ['Group']]
#for i, f in enumerate(prob_cols):
#    sub[f] = pre[:, i]
#sub = sub.drop_duplicates()
#sub.to_csv("./sub/sub.csv",index=False)


###############################################################################

'''
每个预测文件中，
取出相同Group的组进行比较，
把最大值最大的那个组作为最后的预测结果
'''

#submit = pd.read_csv('./data/submit_example2.csv')
#
#path = './em/'
#csv_name = os.listdir(path)
#print(csv_name)
#
#sub = []
#for i in tqdm.tqdm(range(120)):
#    for j in range(len(csv_name)):    
#        df = pd.read_csv(path+csv_name[j])
#        
#        if j == 0:
#            max_m = df.iloc[i, 1:].values
#        else:
#            a_a = df.iloc[i, 1:].values
##            if (max_m.max() - max_m.min()) < (a_a.max() - a_a.min()):
##                max_m = a_a
#            
#            if max_m.max() < a_a.max():
#                max_m = a_a
#    
#    sub.append(max_m)
#
#sub = np.array(sub)
#submit.iloc[:, 1:] = sub
#submit.to_csv('./last/2.csv', index=False)



def em(path, outpath):
    submit = pd.read_csv('./data/submit_example2.csv')

    csv_name = os.listdir(path)
    print(csv_name)
    
    sub = []
    for i in tqdm.tqdm(range(120)):
        for j in range(len(csv_name)):    
            df = pd.read_csv(path+csv_name[j])
            
            if j == 0:
                max_m = df.iloc[i, 1:].values
            else:
                a_a = df.iloc[i, 1:].values
    #            if (max_m.max() - max_m.min()) < (a_a.max() - a_a.min()):
    #                max_m = a_a
                
                if max_m.max() < a_a.max():
                    max_m = a_a
        
        sub.append(max_m)
    
    sub = np.array(sub)
    submit.iloc[:, 1:] = sub
    submit.to_csv(outpath, index=False)
    
em('./em/', 'e:/1.csv')

###############################################################################

'''
线性融合
'''

#a = pd.read_csv('./last/1.csv')
#b = pd.read_csv('./last/2.csv')
#
#
#s = a.copy()
#s.iloc[:, 1:] = a.iloc[:, 1:]*0.45 + b.iloc[:, 1:]*0.55
#s.to_csv('./last/LastSub.csv', index=False)

###############################################################################

















