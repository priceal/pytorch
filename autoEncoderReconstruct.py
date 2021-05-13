#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 10:36:09 2021

@author: allen
"""

datalist = [1,11,1003,2222,3401,4343]


fig,ax = plt.subplots(6,2)
for i,img in zip(range(6),datalist):
    
    y_recon = model(x[img]).detach().numpy()
    ax[i,0].imshow( x[img].reshape((5,5)))
    ax[i,1].imshow( y_recon.reshape((5,5)))