# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:23:34 2021

@author: priceal
"""

import sklearn.datasets
import numpy as np

def load_extra_datasets(N):

    return sklearn.datasets.make_gaussian_quantiles(\
            mean=None,cov=0.7,n_samples=N,n_features=2,\
            n_classes=2,shuffle=True,random_state=None)
        
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*sigmoid(-x)

def propagate(x1,params,details=False):
    """
    Performs forward and backward propagation given input data vector and
    a set of connection weights. Input vector is passed through network layers
    and activation functions. The output vector is returned. The partial
    derivatives of the output vector components wrt the weight matrix components
    is calculated using back propagation. These partial derivatives are returned
    in derivative matrices.
    
    Note: all input vectors and weight matrices are given in "extended" form, 
    where bottom row of data vector is row of '1' values to function as bias terms.
    Bottom rows of connection matrices are bias values.

    Parameters
    ----------
    x1 : float array
        (n+1) x m extended data array, n = dimensions of input data vectors
        and m = number of samples
    params : tuple of arrays
        (w1,w2), the exended weight arrays, dim(w1) = n_h x (n_x+1), and
        dim(w2) = n_y x (n_h+1)

    Returns
    -------
    v3 : array
        the output data vector, not extended.
    DyDw2 : array
        1 x .
    TYPE
        DESCRIPTION.

    """
    w1, w2 = params  # unpack nn model extended weight arrays
    m = x1.shape[1]     # number of data points
    
    # calculate feed forward --- the process is matrix multiply extended
    # data vector by extended weights, calculate activation, then add a bias 
    # component to each data vector
    z1 = np.dot(w1,x1)  # equivalent to z1 = np.einsum('ij,jk->ik',w1,x1)
    v2 = sigmoid(z1)   # (n_h+1) x m matrix, the unextended state vector
    x2 = np.vstack((v2, np.ones((1,m))))  # extend the state vector
    z2 = np.dot(w2,x2)  # equivalent to z2 = np.einsum('ij,jk->ik',w2,x2)
    v3 = sigmoid(z2)    # (n_y) x m output matrix (not extended)
    
    # now do the back propagation---
    # first define some values that are used for calculation
    dsigmz2 = dsigmoid(z2)   # the derivative of layer 2 nodes: 1 x M
    dsigmz1 = dsigmoid(z1)   # the derivative of layer 1 nodes: 4 x M
    delta1 = np.identity(1)  # kronecker delta needed in formation of DyDw2: 1 x 1
    
    # this is matrix derivate of output wrt weight, indices are as follows:
    # DyDwN[ i, j, k , l ]:   i = row of weight matrix
    #                         j = column of weight matrix
    #                         k = row of output vector (node #)
    #                         l = data point number
    
#   DyDw2 = np.einsum('1M,11,5M',dsigmaz2,delta1,x2)
    DyDw2 = np.einsum('mj,nm,lj->nlmj',dsigmz2,delta1,x2)

#   DyDw1 = np.einsum('1M,15,4M,3M',dsigmz2,w2,dsigmz1,x1)
    w2ne = w2[:,:-1]  # this preps the non-extended weight matrix
    DyDw1 = np.einsum('mj,mi,ij,lj->ilmj',dsigmz2,w2ne,dsigmz1,x1)

    if details:
        return v3, v2
    else:
        return v3, DyDw1, DyDw2   # returned array does not get extended bias component

def logloss(y,p):
    """
    
    Parameters
    ----------
    y : TYPE
        the training data
    p : TYPE
        the prediction

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    return - np.log(p)*y - (1.0-y)*np.log(1.0-p)


def dlogloss(y,p):
    """
    

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """ 
    return - y/p + (1-y)/(1-p)