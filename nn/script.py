# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""

xrange = [-2,2]
yrange = [-2,2]
numPoints = 40

gridpoints = np.array(
   [ [x,y] for x in np.linspace(xrange[0],xrange[1],numPoints) \
    for y in np.linspace(yrange[0],yrange[1],numPoints) ] )

plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Spectral)

xinput = np.vstack((gridpoints.T, np.ones((1,numPoints*numPoints))))

predictions,v2 = nn.propagate(xinput,[W1,W2],details=True)

plt.figure(2)
plt.scatter(gridpoints[:,0],gridpoints[:,1],c=np.round(predictions),cmap=plt.cm.Spectral)

for i in range(20):
    plt.figure(3+i)
    plt.scatter(gridpoints[:,0],gridpoints[:,1],c=v2[i],cmap=plt.cm.Spectral)