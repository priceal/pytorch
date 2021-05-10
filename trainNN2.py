# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal
"""
MacroCycles = 10
InnerCycles = 1000

learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for tt in range(MacroCycles):
    for t in range(InnerCycles):
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(tt*InnerCycles,loss.item())

