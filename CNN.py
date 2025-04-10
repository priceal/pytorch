#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:24:09 2025

@author: allen

create an example data set of linearly separable points,
then fit a logistic model to the data using pytorch,
perform a manual optimization with steepest descent
using log loss function ( y = (0,1) p = pred. prob )

L(y,p) = -[ ylog(p) + (1-y)log(1-p) ]

data points chosen randomly in normal dist. about origin.
separating plane normal chosen randomly and displaced from origin
randomly
"""













import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix


'''
###############################################################################
######################### functions ###########################################
###############################################################################
'''

def loadSequence( file, Directory='.')
    
    if Directory:         # use directory if given
        file=os.path.join(Directory,file) 
            
    tempStructure=np.load(file)    
    coords = []  # to receive coords from groups from one file
    for g in group:
        coords.append(tempStructure[g])
        
    # concatenate along residue atom number axis (1) and append
    listCoords.append(np.concatenate(coords,axis=1))
    listSeq.append(tempStructure['seq']) 
        
    return np.concatenate(listCoords,axis=0), \
           np.concatenate(listSeq,axis=0)
           
def loadStructureAndSequence( files, group, Directory='' ):
    
    listCoords = []  # to receive all coords from list of files
    listSeq = []  # to receive sequence from files
    for f in files:   
        if Directory:                   # use directory if given
            f=os.path.join(Directory,f) 
            
        tempStructure=np.load(f)    
        coords = []  # to receive coords from groups from one file
        for g in group:
            coords.append(tempStructure[g])
            
        # concatenate along residue atom number axis (1) and append
        listCoords.append(np.concatenate(coords,axis=1))
        listSeq.append(tempStructure['seq']) 
        
    return np.concatenate(listCoords,axis=0), \
           np.concatenate(listSeq,axis=0)
           
###############################################################################

'''
###############################################################################
############################# main ############################################
###############################################################################
'''

if __name__ == "__main__":
    '''
    structure file information --- format of dictionary entries:
        label : [ [files], {groups} ]
    
    label = used to label axes
    [files] = list of .npz files for this set of structure
    {groups} = set of subgroups of atoms to consider
    allowed groups ...
               proteins:   bb = backbone, sc = sidechain
               dna:        ph = phosphate, rb = ribose, ba = base
    
    N.B. all .npz files under same label MUST have structure arrays of same shape!           
    '''

    # file to load
    inputFile = 
    # optional structure file directory, can leave undefined '' or '.'
    fileDirectory = '../DATA/PDB/npz'
    
    cutoff = 0  # non-zero for a contact map with cutoff value
    mapTitle = 'sidechain-base contacts'
    colorMap = 'OrRd'
    logorithmic = True    
    
 
    





# setup the model/learning parameters
batchSize = 500
dataRegionScale = 2.0    # spread in points
biasSize = 0.5       # perp distance of separating plane from origin
noiseLevel = 0.2
LearningRate = 0.001
numberIterations = 30
reportCycle = 2

# define normal to separating plane and bias - using gaussian trick
# for spherically distributed direction
normal = np.random.normal(scale=1.0,size=dimensions)
normal = normal/np.sqrt((normal*normal).sum())
bias = np.random.normal(scale=biasSize)
print('hidden values:')
print(f'{normal = }\n{bias = }')

# define some sample data - normally distributed
xData = np.random.normal(scale=dataRegionScale,size=(batchSize,dimensions))
yData = np.dot( xData, normal ) > bias  # classify by side of separating plane
yData = yData[:,np.newaxis]
xData += np.random.normal(scale=noiseLevel,size=(batchSize,dimensions)) 

# display sample data
plt.figure(0)
plt.scatter(xData[:,0],xData[:,1],c=yData)
xp = np.array([xData[:,0].min(),xData[:,0].max()])
yp = -xp*normal[0]/normal[1] + bias/normal[1]
plt.plot(xp,yp,'-')
plt.xlim(1.1*xData[:,0].min(),1.1*xData[:,0].max())
plt.ylim(1.1*xData[:,1].min(),1.1*xData[:,1].max())
plt.grid()

# convert to tensors for training
xData = torch.tensor( xData, dtype=torch.float32, requires_grad=True )
yData = torch.tensor( yData, dtype=torch.float32, requires_grad=False)

# define model ###############################################################
class logisticModel(torch.nn.Module):

    def __init__(self,dims):
        super(logisticModel, self).__init__()
        self.layer1 = torch.nn.Linear(dims, 1)
        torch.nn.init.zeros_(self.layer1.bias)
        torch.nn.init.ones_(self.layer1.weight) 
       
    def forward(self, x):
        ypred = torch.sigmoid(self.layer1(x))
       
        return ypred
###############################################################################   
model = logisticModel(dimensions)

# print initial parameters, and display separator
print('initial model ')
for p in model.parameters():
    print(p)
m = model.layer1.weight.detach().numpy()[0]
b = model.layer1.bias.item()
yp = -xp*m[0]/m[1] + b/m[1]
plt.plot(xp,yp,'--')

# run cycles of optimization
#optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate)
#lossfn = torch.nn.CrossEntropyLoss()
plt.figure(1)
for i in range(numberIterations):
    prediction = model(xData)
    lossTerms = -yData*torch.log(prediction)-(1.0-yData)*torch.log(1.0-prediction) 
    loss = lossTerms.sum()
    if i%reportCycle == 0:
        print(f'{loss = }')
        plt.plot([i],[loss.detach().item()],'.k')

    # do manual gradient descent
    loss.backward()
    with torch.no_grad():
        '''
        model.layer1.weight -= model.layer1.weight.grad*LearningRate
        model.layer1.weight.grad.zero_()
        '''
        for p in model.parameters():
            p -= p.grad*LearningRate
            p.grad.zero_()


# print final parameters and display separator
print('final model: ')
for p in model.parameters():
    print(p)
m = model.layer1.weight.detach().numpy()[0]
b = model.layer1.bias.item()
plt.figure(0)
yp = -xp*m[0]/m[1] - b/m[1]
plt.plot(xp,yp,'-')

# metrics
cm = confusion_matrix(yData.detach().numpy(), \
       np.round( prediction.detach().numpy()) ) 
print('confusion matrix:')    
print(f'{' ':>10} {'0 pred':>10} {'1 pred':>10} {'true':>10}')
print(f'{'0 true':>10} {cm[0,0]:>10} {cm[0,1]:>10} {cm[0,0]+cm[0,1]:>10}')
print(f'{'1 true':>10} {cm[1,0]:>10} {cm[1,1]:>10} {cm[1,0]+cm[1,1]:>10}')
print(f'{'pred':>10} {cm[1,0]+cm[0,0]:>10} {cm[1,1]+cm[0,1]:>10} {cm[1,0]+cm[0,0]+cm[1,1]+cm[0,1]:>10}')
print('\n')
print(f'precision = {cm[1,1]/(cm[1,1]+cm[0,1]):.4}')
print(f'recall = {cm[1,1]/(cm[1,1]+cm[1,0]):.4}')


