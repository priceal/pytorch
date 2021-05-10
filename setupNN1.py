# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""

numPoints = 4000
testSetSize = 0.35


# create random data, not linearly separable
X, Y = load_extra_datasets(numPoints)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Spectral)


# split data into training and test set using scikit-learn tools
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=testSetSize, random_state=1)

# initialize nn with random weights
# number of samples, input dims, hidden layer, output
N, D_in, H, D_out = numPoints, 2, 2, 1
x = torch.FloatTensor(X_train)
y = torch.FloatTensor(Y_train[:,np.newaxis]) #torch format
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H,D_out),
    torch.nn.Sigmoid(),
    )
loss_fn = torch.nn.BCELoss()

