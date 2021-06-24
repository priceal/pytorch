# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""

data_file = '../training-data/nearMissData.pkl'

testSetSize = 0.2


# this is a CNN that is intended to reduce input receptive field to a single
# pixel. NC0 = input channels, NC
# number of channels/kernal size in each convolutional layer
NC0, NC1, NC2, NC3 = 1, 4, 16, 1
K1, K2, K3 = 3, 3, 3

##############################################################################
##############################################################################
#load and flatten data
with open(data_file,'rb') as filename:
    frames, classification, coordinates = pickle.load(filename)
numPoints, xdim, ydim = frames.shape
X = frames[:,np.newaxis,:,:]
Y = classification[:,np.newaxis,np.newaxis]

# normalize/scale data
X = X/X.max()

# split into train/test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=testSetSize, random_state=1)

x = torch.FloatTensor(X_train)
y = torch.FloatTensor(Y_train[:,np.newaxis])

model = torch.nn.Sequential(
    nn.Conv2d(NC0,NC1,kernel_size=K1,stride=1,padding=0), 
    nn.Sigmoid(),
    nn.Conv2d(NC1,NC2,kernel_size=K2,stride=1,padding=0),
    nn.Sigmoid(),
    nn.Conv2d(NC2,NC3,kernel_size=K3,stride=1,padding=0),
    nn.Sigmoid(),
    )
loss_fn = torch.nn.BCELoss()