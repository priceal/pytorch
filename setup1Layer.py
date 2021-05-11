# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""



# initialize nn with random weights
# number of samples, input dims, hidden layer, output
D_in, D_out = 2, 1
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,D_out),
    torch.nn.Sigmoid(),
    )


sd = model.state_dict()

for p in sd.keys():
    print(p,sd[p])


sd['0.weight'] = torch.tensor([[2,3]])
sd['0.bias'] = torch.tensor([1])

model.load_state_dict(sd)

for p in sd.keys():
    print(p,sd[p])
