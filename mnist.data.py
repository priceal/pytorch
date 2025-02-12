#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:15:29 2021

@author: allen


this will load in and flatten.....

"""

transform = transforms.Compose([transforms.ToTensor(),
             transforms.Lambda(lambda x: torch.flatten(x))] )
             
trainset = datasets.FashionMNIST('data/MNIST_fashion', train = True, transform = transform)
testset = datasets.FashionMNIST('data/MNIST_fashion', train = False, transform = transform)

trainloader = DataLoader(trainset,batch_size=100,shuffle=True)
testloader = DataLoader(testset,batch_size=100,shuffle=True)

