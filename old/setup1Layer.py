# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal
"""


xMin, xMax = -10, 10



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


sd['0.weight'] = torch.tensor([[5,0]])
sd['0.bias'] = torch.tensor([0])

model.load_state_dict(sd)

for p in sd.keys():
    print(p,sd[p])

sd = model.state_dict()

wx,wy = sd['0.weight'][0].detach().numpy()
bias = sd['0.bias'].detach().numpy()[0]

slope = -wx/wy
intercept = -bias/wy

yLeft = intercept + slope*xMin
yRight = intercept + slope*xMax

print("intercept:", intercept)
print("slope: ", slope)
fig,ax=plt.subplots(2)
ax[0].set_xlim((xMin,xMax))
ax[0].set_ylim((xMin,xMax))
ax[0].set_aspect('equal')
ax[0].plot([xMin,xMax],[yLeft,yRight])
ax[0].grid()

xx = np.linspace(xMin,xMax,100)
a,b = np.meshgrid(xx,xx)
X = np.array(list(zip(a.flatten(),b.flatten())))
x = torch.tensor(X,dtype=torch.float32)
y = model(x)
Y = y.detach().numpy()
YY = Y.reshape((100,100))
ax[1].imshow(YY,origin='lower')

