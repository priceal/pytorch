#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:51:04 2021

@author: allen
"""

# fully connected NN

D_in, H1, H2, H3, D_out = 784, 256, 128, 64, 10

model = nn.Sequential(
    nn.Linear(D_in,H1),
    nn.ReLU(),
    nn.Linear(H1,H2),
    nn.ReLU(),
    nn.Linear(H2,H3),
    nn.ReLU(),
    nn.Linear(H3, D_out),
    nn.LogSoftmax(dim=1),
    )
loss_fn = nn.NLLLoss()
