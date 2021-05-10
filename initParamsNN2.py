#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:53:16 2021

@author: allen
"""



sd = model.state_dict()

for p in sd.keys():
    print(p,sd[p])


scale = 1.0/150.0
a = [-10]*5
b = [0] * 5
c = [20] * 5

h0 = [a,b,c,b,a]

hoa = scale*np.array(h0,dtype=float)
voa = hoa.T


sd['0.weight'] = torch.tensor([hoa.flatten(),voa.flatten()])
sd['0.bias'] = torch.tensor([0,0])

sd['2.weight'] = torch.tensor([[0.5,0.5]])
sd['2.bias'] = torch.tensor([1.0])

model.load_state_dict(sd)

for p in sd.keys():
    print(p,sd[p])