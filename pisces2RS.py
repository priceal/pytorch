#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 10:13:16 2025

@author: allen
"""

import pandas as pd
from sklearn.model_selection import train_test_split

inputFile = 'data/2018-06-06-pdb-intersect-pisces.csv'

minLength = 100
maxLength = 400
testSize = 0.15

outputTest = 'data/pisces100to400.test.txt'
outputTrain = 'data/pisces100to400.train.txt'

####################################################################
df = pd.read_csv(inputFile)

data=[]
for entry in df.itertuples():

    if entry.seq.count('*')>0:      # reject non-standard AAs
        continue
    if len(entry.seq)<minLength:   # reject less than minLength
        continue
    if len(entry.seq)>maxLength:   # reject greater than maxLength
        continue
    data.append(entry.seq+'+'+entry.sst3)

train, test = train_test_split(data, test_size=testSize)

with open(outputTrain,'w') as f:   
    for line in train:
        d, l = line.split('+') 
        f.write( d+'\n' )
        f.write( l+'\n' )
        
with open(outputTest,'w') as f:   
    for line in test:
        d, l = line.split('+') 
        f.write( d+'\n' )
        f.write( l+'\n' )
        
         