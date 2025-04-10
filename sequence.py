#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:24:09 2025

@author: allen




"""

import os
import numpy as np
#import matplotlib.pyplot as plt
import torch
#from sklearn.metrics import confusion_matrix


'''
###############################################################################
######################### functions ###########################################
###############################################################################
'''

def loadSequence( loadFile, Directory='.'):
    
    # use directory if given     
    if Directory:    loadFile=os.path.join(Directory,loadFile)    
    return np.load(loadFile)['seq']
           
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
    '''

    # file to load
    file = '1qpz_A.npz'
    # optional structure file directory, can leave undefined '' or '.'
    directory = '../DATA/PDB/npz'
    
    embeddingDim = 4
    sequenceLength = 20
    
    seq = loadSequence(file,Directory=directory)
    
    embedding = np.random.rand(4,20)
    
    embeddedSeq2 = np.matmul(seq,embedding.T)



    # setup the model/learning parameters
    batchSize = 500
    dataRegionScale = 2.0    # spread in points
    biasSize = 0.5       # perp distance of separating plane from origin
    noiseLevel = 0.2
    LearningRate = 0.001
    numberIterations = 30
    reportCycle = 2

####