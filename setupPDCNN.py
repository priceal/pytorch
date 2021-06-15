# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""

# number of channels/kernal size in each convolutional layer
NC0, NC1, NC2 = 1, 16, 4
K1, K2 = 3, 3

# let's assume an input image of 7 x 7, w/o padding. 
# input layer (0):  1 x 7 x 7
# hidden layer (1): 16 x 5 x 5
# hidden layer (2): 4 x 3 x 3
# output layer (3): 1
# FC layer will be 4x3x3=36 -> 1   
H3, D_out = 36, 1


model = torch.nn.Sequential(
    nn.Conv2d(NC0,NC1,kernel_size=K1,stride=1,padding=0), 
    nn.Sigmoid(),
    nn.Conv2d(NC1,NC2,kernel_size=K2,stride=1,padding=0),
    nn.Sigmoid(),
    nn.Conv2d(NC2,1,kernel_size=3,stride=1,padding=0),
    nn.Sigmoid(),
    )
loss_fn = torch.nn.BCELoss()

##############################################################################

