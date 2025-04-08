# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:36:50 2021

@author: priceal
"""



scanImage = image
xBuffer, yBuffer = 3, 3
scanModel = model
stride = 2

###############################################################################
xDim, yDim = 2*xBuffer+1, 2*yBuffer+1
yFrame, xFrame = scanImage.shape

scaledImage = scanImage/scanImage.max()

mapout = np.zeros((yFrame-2*yBuffer,xFrame-2*xBuffer))
for j in range(0,yFrame-yDim+1,stride):
    if j % 200 == 0:
        print("row",j)
    for i in range(0,xFrame-xDim+1,stride):
        if j % 200 == 0 and i % 200 == 0:
            print("    column",i)
        Xf = scaledImage[j:j+xDim,i:i+yDim].copy().flatten()[np.newaxis,:]
        Xt = torch.FloatTensor(Xf)
        mapout[j,i] = model(Xt).detach().numpy()


