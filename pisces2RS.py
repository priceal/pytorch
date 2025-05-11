#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 10:13:16 2025

@author: allen
"""

import pandas as pd

inputFile = 'data/2018-06-06-pdb-intersect-pisces.csv'
outputFile = 'data/pisces150to250.data.txt'
minLength=150
maxLength=250

df = pd.read_csv(inputFile)

print(df.columns)

count = 0
with open(outputFile,'w') as f:
    lengths=[]
    for entry in df.itertuples():
    
        if entry.seq.count('*')>0:
            continue
        #print(entry.seq)
        #print(entry.sst3)
        if len(entry.seq)<minLength:
            continue
        if len(entry.seq)>maxLength:
            continue
        lengths.append(len(entry.seq))
        f.write(entry.seq+'\n')
        f.write(entry.sst3+'\n')
        count += 1
print(count)

