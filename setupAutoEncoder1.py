# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal

Sets up autoencoder NN 
Intended for train/test set of particle images.

"""

# define parameters
slope = 0.85
intercept = -25
exponent = 1.7
xMin, xMax = 10.0, 100.0
sigma = 1.0

# input dims, hidden layer, output
numberSamples = 500
D_in, H1, H2, H3, D_out = 2, 4, 1, 4, 2
testSetSize = 0.35

#############################################################################
# create the data set
x0 = np.random.uniform(xMin, xMax, numberSamples)
x1 = np.power(x0,exponent)*slope + intercept
x0 += np.random.normal(0, sigma, numberSamples)
x1 += np.random.normal(0, sigma, numberSamples)
X = np.array([x0,x1]).T
Y = X
yMin = x1.min()
yMax = x1.max()
fig0,ax0 = plt.subplots(1)
ax0.set_xlim([xMin, xMax])
ax0.set_ylim([yMin, yMax])
ax0.scatter(X[:,0],X[:,1])

# split data into training and test set using scikit-learn tools
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=testSetSize, random_state=1)

# define training tensors
x = torch.FloatTensor(X_train)
y = torch.FloatTensor(Y_train)
x_all = torch.FloatTensor(X)

# define model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.LeakyReLU(0.5),
    torch.nn.Linear(H1, H2),
    torch.nn.LeakyReLU(0.5),
    torch.nn.Linear(H2, H3),
    torch.nn.LeakyReLU(0.5),
    torch.nn.Linear(H3, D_out),
    )
loss_fn = torch.nn.MSELoss()
print(model.state_dict())



