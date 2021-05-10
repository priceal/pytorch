# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:19:11 2021

@author: priceal
"""
# set up some boolean arrays with 
m = len(Y)
YPOS = Y == 1
y_preda = y_pred.detach().numpy()
PPOS = np.round(y_preda[:,0]) == 1

actualHits = Y.sum()
actualMisses = m-actualHits

predHits = PPOS.sum()
predMisses = m-predHits

truePositives = (PPOS & YPOS).sum()
falsePositives = (PPOS & ~YPOS).sum()
trueNegatives = (~PPOS & ~YPOS).sum()
falseNegatives = (~PPOS & YPOS).sum()

print('')
print('test set contains {} positive and {} negatives'.format(actualHits,actualMisses))
print('prediction set contains {} positive and {} negatives'.format(predHits,predMisses))
print('')
print('true positives', truePositives, '({:2.1f}%)'.format(100*truePositives/actualHits))
print('false negatives', falseNegatives, '({:2.1f}%)'.format(100*falseNegatives/actualHits))
print('')

print('true negatives', trueNegatives, '({:2.1f}%)'.format(100*trueNegatives/actualMisses))
print('false positives', falsePositives, '({:2.1f}%)'.format(100*falsePositives/actualMisses))

