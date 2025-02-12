# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal
"""


totals = 0
datacount = 0
for images, labels in testloader:
    predictions = model(images).detach().numpy().argmax(1)
    corrects = predictions == labels.detach().numpy()
    totals += corrects.sum()
    datacount += len(labels)
print(totals, datacount, totals/datacount)
    
    

