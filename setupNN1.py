# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""

# create random data, not linearly separable
X, Y = load_extra_datasets(500)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Spectral)


# split data into training and test set


# initialize nn with random weights
# number of samples, input dims, hidden layer, output
N, D_in, H, D_out = 500, 2, 4, 1
x = torch.FloatTensor(X)
y = torch.FloatTensor(Y[:,np.newaxis])
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H,D_out),
    torch.nn.Sigmoid(),
    )
loss_fn = torch.nn.BCELoss()

