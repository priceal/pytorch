# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""
with open('5x5.pkl','rb') as filename:
    X, Y = pickle.load(filename)

X = X.reshape(5000,25)/X.flatten().max()

# number of samples, input dims, hidden layer, output
N, D_in, H, D_out = 5000, 25, 2, 1
x = torch.FloatTensor(X)
y = torch.FloatTensor(Y[:,np.newaxis])
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H,D_out),
    torch.nn.Sigmoid(),
    )
loss_fn = torch.nn.BCELoss()

