#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:59:13 2021

@author: allen
"""

data_file = '../training-data/data.pkl'
testSetSize = 0.2

batchSize = 500

##############################################################################
#load and flatten data
with open(data_file,'rb') as filename:
    frames, classification, coordinates = pickle.load(filename)
numPoints, xdim, ydim = frames.shape
# define X, Y and insert axes for channels and pxl coords
# the axes are: SampleNumber, Channel, y-coord, x-coord
X = frames[:,np.newaxis,:,:] 
Y = classification[:,np.newaxis,np.newaxis,np.newaxis]

# normalize/scale data
X = X/X.max()

# split into train/test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=testSetSize, random_state=1)


# now create the data loader for the training set
xt = torch.FloatTensor(X_train)
yt = torch.FloatTensor(Y_train)
training_list = list(zip(xt,yt))
trainLoader = DataLoader(training_list,batch_size=batchSize,shuffle=True)


# number of samples, input dims, hidden layer, output
#x = torch.FloatTensor(X_train)
#y = torch.FloatTensor(Y_train)

# load data into loader

#DataLoader(
#    dataset,
#    batch_size=1,
#   shuffle=False,
#    num_workers=0,
#    collate_fn=None,
#    pin_memory=False,
# )
