# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:19:11 2021

@author: priceal
"""
# define data to evaluate model, must have X and Y values
x_eval =  X_test
y_eval = Y_test[:,0,0] # remove redundant axes (CNN case)

# name
eval_name = 'test set'

#############################################################################
# set up some boolean arrays with 
m = len(y_eval)
YPOS = y_eval == 1
x_tens = torch.FloatTensor(x_eval)
y_pred_eval = model(x_tens)
y_preda = y_pred_eval.detach().numpy()[:,0,0,0]
PPOS = np.round(y_preda) == 1  

# define some stats
actualHits = y_eval.sum()
actualMisses = m-actualHits
predHits = PPOS.sum()
predMisses = m-predHits
#truePositives = (PPOS & YPOS).sum()
#falsePositives = (PPOS & ~YPOS).sum()
#trueNegatives = (~PPOS & ~YPOS).sum()
#falseNegatives = (~PPOS & YPOS).sum()

cf = confusion_matrix(y_eval,PPOS)
truePositives = cf[1,1]
falsePositives = cf[0,1]
trueNegatives = cf[0,0]
falseNegatives = cf[1,0]
negatives = trueNegatives + falseNegatives
positives = truePositives + falsePositives

print('')
print('test set contains {} TRUE and {} FALSE'.format(actualHits,actualMisses))
print('predictions contain {} POSITIVE and {} NEGATIVE'.format(predHits,predMisses))
print('')

print('recall {:2.1f}%'.format(100*truePositives/actualHits) )
print('precision {:2.1f}%'.format(100*truePositives/(truePositives+falsePositives)) )

print('')
print('           CONFUSION MATRIX')
print('           FALSE      TRUE       TOTAL')
print('NEGATIVE   {:<10} {:<10} {:<10}'.format(trueNegatives,falseNegatives,negatives))
print('POSITIVE   {:<10} {:<10} {:<10}'.format(falsePositives,truePositives,positives))
print('TOTAL      {:<10} {:<10} {:<10}'.format(actualMisses,actualHits,m))


