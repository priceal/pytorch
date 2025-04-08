# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal
"""

# define number of reporting macrocycles and epochs per reporting cycle
epochs = 30
reporting = 1

# optimization parameters
learning_rate = 0.01

##############################################################################
optimizer = optim.SGD(model.parameters(),lr=learning_rate)

lossHistory = []
for t in range(epochs):
    totalLoss = 0
    for images, labels in trainloader:
        labels_pred = model(images)
        loss = loss_fn(labels_pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
    lossHistory.append(totalLoss)
    if t % reporting == 0 :
        print(t,totalLoss)
plt.plot(lossHistory)
    

