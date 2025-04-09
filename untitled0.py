#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:19:13 2025

@author: allen
"""
import torch.nn as nn
import torch

# Example of target with class indices
loss = nn.CrossEntropyLoss()
inpt = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(inpt, target)
output.backward()

# Example of target with class probabilities
inpt = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(inpt, target)
output.backward()