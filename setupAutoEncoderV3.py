# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal

Sets up autoencoder NN to map images onto images.
Intended for train/test set of particle images.

able to get to MSEL < 0.0007 with 2 dims representation! compare to 
linear rep with 4 dims, = 0.005
with fewer dimensions in internal latent rep, get better MSEL !!!

but needed 5 hidden layers! 

"""

data_file = '5x5.pkl'

# input dims, hidden layer, output
D_in, H1, H2, H3, H4, H5, D_out = 25, 8, 4, 2, 4, 8, 25

testSetSize = 0.25

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
xt = torch.FloatTensor(X_test)
yt = torch.FloatTensor(Y_test)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H1),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H1,H2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H2,H3),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H3,H4),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H4,H5),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H5,D_out),
    torch.nn.Sigmoid(),
    )
loss_fn = torch.nn.MSELoss()

#print(model.state_dict())



