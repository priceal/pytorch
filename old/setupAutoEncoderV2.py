# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal

Sets up autoencoder NN to map images onto images.
Intended for train/test set of particle images.

can go down to MSLoss = 0.0006 with internal layer rep of 4 dims
better than linear model!!

"""

data_file = '5x5.pkl'

# input dims, hidden layer, output
D_in, H1, H2, H3, D_out = 25, 8, 4, 8, 25

testSetSize = 0.01

##############################################################################
with open(data_file,'rb') as filename:
    X, junk = pickle.load(filename)
numPoints = len(X)
X = X.reshape(numPoints,25)/X.flatten().max()
Y = np.copy(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=testSetSize, random_state=1)
x = torch.FloatTensor(X_train)
y = torch.FloatTensor(Y_train)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H1),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H1,H2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H2,H3),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H3,D_out),
    torch.nn.Sigmoid(),
    )
loss_fn = torch.nn.MSELoss()

print(model.state_dict())



