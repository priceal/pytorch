#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:15:29 2021

@author: allen
"""

images, labels = next(iter(trainloader))
preds = model(images).detach().numpy().argmax(1)
fig,ax = plt.subplots(5,5)
axf = ax.flatten()
for i, axn in enumerate(axf):
    axn.imshow(images[i].reshape((28,28)))
    axn.set_title(str(labels[i].detach().numpy()) + '/' + str(preds[i] ))
    axn.set_axis_off()
    
