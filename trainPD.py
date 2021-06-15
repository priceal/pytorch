# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal
"""

# define number of reporting macrocycles and epochs per reporting cycle
epochs = 100
reporting = 10

# optimization parameters
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for tt in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if tt % reporting == 0:
        print(tt,loss.item())
    
