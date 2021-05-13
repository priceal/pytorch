# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""


xMin, xMax = -10, 10

# initialize nn with random weights
# number of samples, input dims, hidden layer, output
D_in, H, D_out = 2, 2, 1
model = torch.nn.Sequential(
    torch.nn.Linear(D_in,H),
    torch.nn.Sigmoid(),
    torch.nn.Linear(H,D_out),
    torch.nn.Sigmoid(),
    )

sd = model.state_dict()
for p in sd.keys():
    print(p,sd[p])

sd['0.weight'] = torch.tensor([[0,20],[20,0]])
sd['0.bias'] = torch.tensor([0,0])

sd['2.weight'] = torch.tensor([[2,2]])
sd['2.bias'] = torch.tensor([-1.9])

model.load_state_dict(sd)
for p in sd.keys():
    print(p,sd[p])

sd = model.state_dict()
wx,wy = sd['0.weight'][0].detach().numpy()
bias = sd['0.bias'].detach().numpy()[0]

slope = -wx/wy
intercept = -bias/wy
print("intercept:", intercept)
print("slope: ", slope)
fig,ax=plt.subplots(1)

xx = np.linspace(xMin,xMax,100)
a,b = np.meshgrid(xx,xx)
X = np.array(list(zip(a.flatten(),b.flatten())))
x = torch.tensor(X,dtype=torch.float32)
y = model(x)
Y = y.detach().numpy()
YY = Y.reshape((100,100))
ax.imshow(YY,origin='lower')

