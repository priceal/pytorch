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
exponent = 2.0
xMin, xMax = 10.0, 100.0
sigma = 1.0

# input dims, hidden layer, output
numberSamples = 500
D_in, H, D_out = 4, 1, 2
testSetSize = 0.20

#############################################################################
# create the data set using features/attributes
x0 = np.random.uniform(xMin, xMax, numberSamples)
x1 = np.power(x0,exponent)*slope + intercept
x0 += np.random.normal(0, sigma, numberSamples)
x1 += np.random.normal(0, sigma, numberSamples)
X = np.array([x0,x1,x0*x0,x1*x1]).T
Y = X[:,:2]
yMin = x1.min()
yMax = x1.max()
fig0,ax0 = plt.subplots(1)
#ax0.set_xlim([xMin, xMax])
#ax0.set_ylim([yMin, yMax])
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
    torch.nn.Linear(D_in, H),
    torch.nn.Linear(H, D_out),
    )
loss_fn = torch.nn.MSELoss()
print(model.state_dict())



