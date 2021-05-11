# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""

X, Y = nn.load_extra_datasets(500)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Spectral)

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X,Y)

Ypred = clf.predict(X)
