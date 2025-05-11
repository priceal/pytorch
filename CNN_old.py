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
from Bio import SeqIO

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
       
    return seqsOneHot, classesOneHot
        
def loadSequence( file, Directory='.'):
    
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
           
           
def is_protein( chain ):
    '''
    determines if at least 1 residue in chain is canonical AA residue type 
    which does not code for a DNA/RNA base. Uses a set 'trick'

    Args:
        chain (str): DESCRIPTION.

    Returns:
        bool: DESCRIPTION.

    '''
    return bool( set(chain).intersection( set('RDEQHLKMFPSWYV') ) )


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

        self.layer3 = torch.nn.Conv1d(101, 7, 11, padding='same')
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
    
'''    
    read = SeqIO.parse(os.path.join(fileDirectory,inputFile),'fasta')
    for record in read:
     print(f'{record.id} length = {len(record)}')    
     print(repr(record.seq))
     if is_protein( str(record.seq) ):
         # terminate id at | -- simplifies id in clustering
         print(f'    ....protein! ')
         seq = str(record.seq)
'''
