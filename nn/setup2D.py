# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal

use this to create the data set for testing the nn
the data set is a set of 2D vectors with the class chosen as the central
portion inside a circle (not linearly separable)

run scripts in this order:
    
setup2D.py

train2D.py



"""

X, Y = nn.load_extra_datasets(2000)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Spectral)
