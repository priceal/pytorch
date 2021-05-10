# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""

data_file = '5x5.pkl'


testSetSize = 0.35



##############################################################################
with open(data_file,'rb') as filename:
    X, Y = pickle.load(filename)

numPoints = len(Y)
X = X.reshape(numPoints,25)/X.flatten().max()



X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=testSetSize, random_state=1)



# number of samples, input dims, hidden layer, output
N, D_in, H, D_out = numPoints, 25, 2, 1
x = torch.FloatTensor(X)
y = torch.FloatTensor(Y[:,np.newaxis])
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H,D_out),
    torch.nn.Sigmoid(),
    )
loss_fn = torch.nn.BCELoss()

print(model.state_dict())