# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:21:40 2021

@author: priceal
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import sklearn.datasets
from sklearn.model_selection import test_train_split
import pylab as plt
import pickle

def load_extra_datasets(N):

    return sklearn.datasets.make_gaussian_quantiles(\
            mean=None,cov=0.7,n_samples=N,n_features=2,\
            n_classes=2,shuffle=True,random_state=None)
        