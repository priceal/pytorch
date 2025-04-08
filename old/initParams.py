#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:53:16 2021

@author: allen
"""



sd = model.state_dict()

for p in sd.keys():
    print(p,sd[p])


#w = [[0,1],[0,1],[1,0],[1,0]]
#b = [-2,2,-2,2]

w = [[0,100]]
b = [-300,300]

sd['0.weight'] = torch.tensor(w)
sd['0.bias'] = torch.tensor(b)

sd['2.weight'] = torch.tensor([[1.0]])
sd['2.bias'] = torch.tensor([0])

model.load_state_dict(sd)

for p in sd.keys():
    print(p,sd[p])