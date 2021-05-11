# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 06:36:03 2021

@author: priceal
"""


a = [-10] * 5

b=[0] * 5
c = [20] * 5

h0 = [a,b,c,b,a]

hoa = np.array(h0,dtype=float)

voa = hoa.T

W1[0,:25] = hoa.flatten()

W1[1,:25] = voa.flatten()

W1[0,25]=0
W1[1,25]=0
W2 = np.array([[10,10,10]])
