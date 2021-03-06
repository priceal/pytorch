# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal
"""

# define number of reporting macrocycles and epochs per reporting cycle
MacroCycles = 2
InnerCycles = 2

# optimization parameters
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# learning epochs
for tt in range(MacroCycles):
    for t in range(InnerCycles):
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(tt*InnerCycles,loss.item())

