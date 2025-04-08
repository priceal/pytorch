# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""

data_file = '../training-data/data.pkl'

testSetSize = 0.2

#define fully connected network structure
H1, H2, D_out = 16, 4, 1
       
##############################################################################
#load and flatten data
with open(data_file,'rb') as filename:
    frames, classification, coordinates = pickle.load(filename)
numPoints, xdim, ydim = frames.shape
X = frames.reshape(numPoints,xdim*ydim)
Y = classification

# normalize/scale data
X = X/X.max()

# split into train/test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=testSetSize, random_state=1)

# number of samples, input dims, hidden layer, output
x = torch.FloatTensor(X_train)
y = torch.FloatTensor(Y_train[:,np.newaxis])
model = torch.nn.Sequential(
    torch.nn.Linear(xdim*ydim, H1),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H1, H2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H2, D_out),
    torch.nn.Sigmoid(),
    )
loss_fn = torch.nn.BCELoss()

