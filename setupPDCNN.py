# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""
data_file = '../training-data/data.pkl'
testSetSize = 0.2

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

# number of samples, input dims, hidden layer, output
x = torch.FloatTensor(X_train)
y = torch.FloatTensor(Y_train)
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

