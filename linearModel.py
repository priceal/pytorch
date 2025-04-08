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
from sklearn.metrics import r2_score, mean_squared_error
import pylab as plt

# setup the model/learning parameters
BATCH_SIZE = 20
DIM_IN = 3
DIM_OUT = 7
noiseLevel = 0.4
LearningRate = 0.02
numberIterations = 4000
reportInterval = 1000

# define hidden matrix/vector
matrix = torch.randn( DIM_OUT, DIM_IN, requires_grad=False)
vector = torch.randn( 1, DIM_OUT, requires_grad=False)
print('hidden values')
print(f'{matrix = }\n{vector = }')

# define some sample data
xData = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
yData= torch.matmul(xData,matrix.T)+vector
yData += torch.tensor( 
    np.random.normal(scale=noiseLevel,size=(BATCH_SIZE,DIM_OUT)) 
    )
# (matrix X someInput.T )T = someInput X matrix.T
print(f'{xData = }\n {yData = }')

# define model
class linearModel(torch.nn.Module):

    def __init__(self):
        super(linearModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, DIM_OUT)
       
    def forward(self, x):
        x = self.layer1(x)
        
        return x
    
model = linearModel()

# store initial parameters
print('initial model ')
p0=[]
for p in model.parameters():
    p0.append(p.detach().numpy() )
print(p0)

# run cycles of optimization
lossTraj=[]
for i in range(numberIterations):
    prediction = model(xData)
    loss = (yData - prediction).pow(2).sum()
    if i%reportInterval ==0:
        print(f'{i} {loss = }')
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

print('metrics')
r2s = r2_score(
    yData.numpy(), model(xData).detach().numpy()
    )
mse = mean_squared_error(
    yData.numpy(), model(xData).detach().numpy()
    )
print(f'MSE = {mse}')
print(f'R2 = {r2s}')