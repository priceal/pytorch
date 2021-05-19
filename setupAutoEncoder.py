# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal

Sets up autoencoder NN to map images onto images.
Intended for train/test set of particle images.



with linear activation, can go to 0.005 w/ 4 dims internal rep!!!
with 4D logistic regr. can go to 0.0015,almost as good as 5 layer NN with 2
latent dims

but logistic regr. w/ 2D levels off at MSLE ~ 0.017 >>> 0.0007 with 2D 5 layer NN !!!
also---reconstructions look much better with the autoencoder!!!

"""

data_file = '5x5.pkl'

# input dims, hidden layer, output
D_in, H, D_out = 25, 2, 25

testSetSize = 0.01

##############################################################################
# load data, normalize to 0-1, and make target = X
with open(data_file,'rb') as filename:
    X, junk = pickle.load(filename)
numPoints = len(X)
X = X.reshape(numPoints,25)/X.flatten().max()
Y = np.copy(X)

# split into test and train data, create training tensors
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=testSetSize, random_state=1)
x = torch.FloatTensor(X_train)
y = torch.FloatTensor(Y_train)

# define model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
#    torch.nn.Sigmoid(),
    torch.nn.Linear(H,D_out),
    torch.nn.Sigmoid(),
    )
loss_fn = torch.nn.MSELoss()

print(model.state_dict())



