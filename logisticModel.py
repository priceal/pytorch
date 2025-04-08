#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:24:09 2025

@author: allen


create an example linear relation and some data,
then fit a linear model to the data using pytorch,
perform a manual optimization with steepest descent

"""

import numpy as np
import torch
import pylab as plt

# setup the model/learning parameters

batchSize = 20
dimensions = 2
dataRegionScale = 2.0
biasSize = 1.0
LearningRate = 0.001
numberIterations = 10

# define normal to separating plane and bias
normal = np.random.normal(scale=1.0,size=dimensions)
normal = normal/np.sqrt((normal*normal).sum())
bias = np.random.normal(scale=biasSize)

print('hidden values')
print(f'{normal = }\n{bias = }')

# define some sample data
xData = np.random.normal(scale=dataRegionScale,size=(batchSize,dimensions))
yData = np.dot( xData, normal ) > bias

'''
for x,y in zip(xData,yData):
    print(f'{x = } {y = }')
    if y:
        plt.plot(x[0],x[1],'ob')
    else:
        plt.plot(x[0],x[1],'sr')
plt.grid()
'''
# convert to tensors for training
xData = torch.tensor( xData, dtype=torch.float32, requires_grad=True )
yData = torch.tensor( yData, dtype=torch.long, requires_grad=False)

# (matrix X someInput.T )T = someInput X matrix.T
print(f'{xData = }\n {yData = }')

# define model
class logisticModel(torch.nn.Module):

    def __init__(self):
        super(logisticModel, self).__init__()

        self.layer1 = torch.nn.Linear(dimensions, 1)
        self.sigmoid = torch.nn.Sigmoid()
       
    def forward(self, x):
        x = self.layer1(x)
        p = self.sigmoid(x) 
        q = self.sigmoid(-x)
        
        return torch.tensor(list(zip(p,q)),requires_grad=True)
    
model = logisticModel()

# store initial parameters
print('initial model ')
p0=[]
for p in model.parameters():
    p0.append(p.detach().numpy() )
print(p0)

# run cycles of optimization
lossTraj=[]
lossFunction = torch.nn.NLLLoss()

for i in range(numberIterations):
    prediction = model(xData)
    loss = lossFunction( prediction, yData )
    print(f'{loss = }')
    loss.backward()
    lossTraj.append(loss.item())

    # do manual gradient descent
    with torch.no_grad():
        for p in model.parameters():
            p -= p.grad*LearningRate
            p.grad.zero_()

# print/store final parameters
print('final model ')
p1=[]
for p in model.parameters():
    p1.append(p.detach().numpy() )
print(p1)
print(f'{matrix = }\n{vector = }')
plt.semilogy(lossTraj)

