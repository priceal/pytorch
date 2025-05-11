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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

'''
###############################################################################
######################### functions ###########################################
###############################################################################
'''

def dataLoader( filePath, maxLen=800 ):
    
    with open( filePath, 'r' ) as f:
        lines = f.readlines()
    sequences = lines[::2]
    classes = lines[1::2]
    
    seqsOneHot = []
    for sequence in sequences:
        sequence = f'{sequence[:800]:<800}'
        seqsOneHot.append(oneHot( sequence ) )
    
    classesOneHot = []
    for clss in classes:
        clss = f'{clss[:800]:<800}'
        result = []
        for c in clss:
            code = np.zeros( 4 )      
            index=" HEC".find(c) 
            code[index] = 1
            result.append(code)
        
        classesOneHot.append( result )
       
    x = np.array(seqsOneHot).swapaxes(1,2)
    y = np.array(classesOneHot).swapaxes(1,2)
    return torch.tensor( x, dtype=torch.float32, requires_grad=True ),\
        torch.tensor( y, dtype=torch.float32, requires_grad=True) 
          

#######################################################################
# create the oneHot array for the sequence
aminoAcids = "ARNDCEQGHILKMFPSTWYV"
def oneHot( string ):
    
    result = []
    for c in string:
        code = np.zeros( 21 )      
        index=" ARNDCEQGHILKMFPSTWYV".find(c) 
        code[index] = 1
        result.append(code)
        
    return np.array(result)

#######################################################################
    
        
###############################################################################



# define model ###############################################################
class cnnModel(torch.nn.Module):

    def __init__(self):
        super(cnnModel, self).__init__()
        
        # input channels = number of AA's +1 for padding. pad so that
        # output same length, padding character = 0
        # (inputs, outputs, kernel)
        self.layer1 = torch.nn.Conv1d(21, 21, 11, padding='same')
        self.relu1 = torch.nn.ReLU()
        
        self.layer2 = torch.nn.Conv1d(21, 101, 101, padding='same')
        self.relu2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Conv1d(101, 4, 11, padding='same')
        self.softMax3 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.layer1(x)
        x = self.relu1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        
        x = self.layer3(x)
        x = self.softMax3(x)
        
        return x
###############################################################################   



'''
###############################################################################
############################# main ############################################
###############################################################################
'''

maxLength = 800
numberIterations = 100
reportCycle = 5
LearningRate = 0.0000001

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
    inputFile = 'RS126.data.txt'
    
    # optional  file directory, can leave undefined '' or '.'
    fileDirectory = 'data'

    x, y = dataLoader( os.path.join(fileDirectory,inputFile) ) 
 
    model = cnnModel()
    
    
    
    # run cycles of optimization
    
    ''' 
    output is of shape (800,4), with probability of each class. training
    data is one hot rep. I imagine what you want is something like 
    
    -yData*torch.log(prediction)-(1.0-yData)*torch.log(1.0-prediction) 
    
    
    '''
    
    
#optimizer = torch.optim.SGD(model.parameters(), lr=LearningRate)
#lossfn = torch.nn.CrossEntropyLoss()
plt.figure(1)
for i in range(numberIterations):
    prediction = model(x)
    lossTerms = -y*torch.log(prediction)-(1.0-y)*torch.log(1.0-prediction) 
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

    

    #  
    
    # metrics
    yCheck = np.argmax(y.detach().numpy(),axis=1).flatten()
    pCheck = np.argmax(prediction.detach().numpy(),axis=1).flatten()
    
    cm = confusion_matrix( yCheck, pCheck ) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, \
                                  display_labels=['_','H','E','C'])
    

    disp.plot()

    plt.show()
 
    
 
    
 
