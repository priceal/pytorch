#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 09:01:29 2025

@author: allen
"""

# %matplotlib inline

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

# let's take function f(x) = sin(x)*y^2
# df/dx = cos(x)*y^2
# df/dy = 2ysin(x)

x = torch.linspace(-1., 1., steps=9, requires_grad=True)
y = torch.linspace(-1., 1., steps=19, requires_grad=True)

f1=torch.sin(x)
f2=torch.pow(y,2)
f=f1*f2

plt.plot(x.detach(),f.detach())

plt.plot(xx,5*xx*xx-3*xx*xx*xx,'o')
plt.plot(xx, 10*xx-9*xx*xx,'s')

out=torch.sum(f)
out.backward()
