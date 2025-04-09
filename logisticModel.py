#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:24:09 2025

@author: allen


create an example data set of linearly separable points,
then fit a logistic model to the data using pytorch,
perform a manual optimization with steepest descent
using log loss function ( y = (0,1) p = pred. prob )

L(y,p) = -[ ylog(p) + (1-y)log(1-p) ]

data points chosen randomly in normal dist. about origin.
separating plane normal chosen randomly and displaced from origin
randomly
"""

import numpy as np
import torch
import pylab as plt

# setup the model/learning parameters
batchSize = 50
dimensions = 2
dataRegionScale = 2.0    # spread in points
biasSize = 0       # perp distance of separating plane from origin
LearningRate = 0.001
numberIterations = 10
reportCycle = 1

# define normal to separating plane and bias - using gaussian trick
# for spherically distributed direction
normal = np.random.normal(scale=1.0,size=dimensions)
normal = normal/np.sqrt((normal*normal).sum())
bias = np.random.normal(scale=biasSize)
print('hidden values:')
print(f'{normal = }\n{bias = }')

# define some sample data - normally distributed
xData = np.random.normal(scale=dataRegionScale,size=(batchSize,dimensions))
yData = np.dot( xData, normal ) > bias  # classify by side of separating plane
yData = yData[:,np.newaxis]
'''
for x,y in zip(xData,yData):
    print(f'{x = } {y = }')
    if y:
        plt.plot(x[0],x[1],'ob')
    else:
        plt.plot(x[0],x[1],'sr')

'''
plt.figure(0)
plt.scatter(xData[:,0],xData[:,1],c=yData)
xp = np.array([xData[:,0].min(),xData[:,0].max()])
yp = -xp*normal[0]/normal[1] + bias/normal[1]
plt.plot(xp,yp,'-')
plt.grid()

# convert to tensors for training
xData = torch.tensor( xData, dtype=torch.float32, requires_grad=True )
yData = torch.tensor( yData, dtype=torch.float32, requires_grad=False)

# (matrix X someInput.T )T = someInput X matrix.T
print(f'{xData = }\n {yData = }')

# define model
class logisticModel(torch.nn.Module):

    def __init__(self):
        super(logisticModel, self).__init__()

        self.layer1 = torch.nn.Linear(dimensions, 1)
          
        torch.nn.init.zeros_(self.layer1.bias)
        torch.nn.init.ones_(self.layer1.weight) 
       
    def forward(self, x):
        ypred = torch.sigmoid(self.layer1(x))
       
        return ypred
    
    
    
    
    
    
model = logisticModel()

# store initial parameters
print('initial model ')
p0=[]
for p in model.parameters():
    p0.append(p.detach().numpy() )
print(p0)
m = model.layer1.weight.detach().numpy()[0]
b = model.layer1.bias.item()
yp = -xp*m[0]/m[1] + b/m[1]
plt.plot(xp,yp,'--')


# run cycles of optimization
lossTraj=[]
optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate)
lossfn = torch.nn.CrossEntropyLoss()

for i in range(numberIterations):
    optimizer.zero_grad()
    prediction = model(xData)
###    lossTerms = -yData*torch.log(prediction)-(1.0-yData)*torch.log(1.0-prediction) 
###    loss = lossTerms.sum()
    loss = lossfn(prediction,yData)
    print(f'{loss = }')
    loss.backward()
    if i%reportCycle == 0:
        lossTraj.append(loss.item())
    optimizer.step()
'''
    # do manual gradient descent
    with torch.no_grad():
        model.layer1.weight -= model.layer1.weight.grad*LearningRate
        model.layer1.weight.grad.zero_()
OR
        for p in model.parameters():
            p -= p.grad*LearningRate
            p.grad.zero_()
'''

# print/store final parameters
print('final model ')
p1=[]
for p in model.parameters():
    p1.append(p.detach().numpy() )
print(p1)
print(normal,bias)

# final solution
m = model.layer1.weight.detach().numpy()[0]
b = model.layer1.bias.item()
plt.figure(0)
yp = -xp*m[0]/m[1] + b/m[1]
plt.plot(xp,yp,'-')


plt.figure(1)
plt.semilogy(lossTraj)

