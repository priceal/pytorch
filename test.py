#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 08:16:28 2025

@author: allen
"""

import torch
import numpy as np
#import seaborn as sns

# 
x = torch.tensor([[1.,2.,3.],[4.,5.,6.]],requires_grad=True)
y = torch.tensor([[10.],[20.]],requires_grad=True)


print(f'{x = }')
print(f'{y = }')

#
a = 2.0*x
b = a*a
sm1 = b.sum()


c = a*y
d = 10*c

sm2 = d.sum()




#sns.heatmap(x.detach(),annot=True)

y = x.sum()
print(f'{y = }')
print(f'{x.requires_grad = }')
print(f'{y.requires_grad = }')
print()
print(f'{x.grad_fn = }')
print(f'{y.grad_fn = }')
print()
print(f'{x.grad = }')
print()
for i in range(3):
    y.backward()
    print(f'{x.grad = }')
