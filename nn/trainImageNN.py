# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:40:23 2021

@author: priceal

this is a NN that functions as a 2D classifier: input is a 2D vector which
is determined to be in a target set or not.

"""

learnRate =     2.0
#macroCycles = 50
#innerCycles = 300

macroCycles = 20
innerCycles = 100
Xset  = norms
Yset = xylist



# training set is defined X = m x n*n array of m samples, Y = m dimensional
# array with classification either 1/0 or True/False
m,dx,dy = len(Yset),5,5      # number of samples
ytrain = Yset[np.newaxis,:].astype(int)
X = Xset
YPOS = ytrain == 1
actualHits = ytrain.sum()
actualMisses = m-actualHits

preFactor = learnRate/m

# add a one (1) to the training data set to represent the bias term
# and transpose so data matrix has data vectors vertically, with samples
# arranged horixontally
x1 = np.vstack((X.T, np.ones((1,m))))


#define number of nodes per layer, this is fixed, but can be changed later
n_x = dx*dy    # input layer (1), the 2D vector
n_h = 2    # hidden layer (2), 4 hidden nodes
n_y = 1    # output layer (3), a single logical output

# there are two weight matrices linking input-hidden, and hidder-output layer
# initialize eight matrices
#W1 = np.random.randn(n_h,n_x+1)*0.1
#W2 = np.random.randn(n_y,n_h+1)*0.1

print(W1)
print(W2)
#print(ytrain)
for macroIteration in range(macroCycles):
    
    for innerIteration in range(innerCycles):
    
        # propagate data vector, returns prediction and derivs. matrices
        model = [W1,W2]      # list contains all parameters
        ypred,dw1,dw2 = nn.propagate(x1,model)
        
        # calculate logloss and logloss derivatives
        ll = nn.logloss(ytrain,ypred)
        dll = nn.dlogloss(ytrain,ypred)

        # use back propagated derivs matrices to calculate matrix derivs
        dCdw1 = np.einsum('ij,nlij->nl',dll,dw1)
        dCdw2 = np.einsum('ij,nlij->nl',dll,dw2)

        # increment W1, W2 down gradient
        W1 = W1 - preFactor*dCdw1
        W2 = W2 - preFactor*dCdw2

    print("cycle: ",(macroIteration+1)*innerCycles, 'total logloss:',ll.sum())     
    PPOS = np.round(ypred) == 1
    predHits = PPOS.sum()
    predMisses = m-predHits
    truePositives = (PPOS & YPOS).sum()
    falsePositives = (PPOS & ~YPOS).sum()
    trueNegatives = (~PPOS & ~YPOS).sum()
    falseNegatives = (~PPOS & YPOS).sum()
  
    print('true positives', truePositives, '({:2.1f}%)'.format(100*truePositives/actualHits))
    print('false negatives', falseNegatives, '({:2.1f}%)'.format(100*falseNegatives/actualHits))
    print('true negatives', trueNegatives, '({:2.1f}%)'.format(100*trueNegatives/actualMisses))
    print('false positives', falsePositives, '({:2.1f}%)'.format(100*falsePositives/actualMisses))
    print('')

print('FINAL STATISTICS')
print('test set contains {} positive and {} negatives'.format(actualHits,actualMisses))
print('prediction set contains {} positive and {} negatives'.format(predHits,predMisses))
print('')
print('true positives', truePositives, '({:2.1f}%)'.format(100*truePositives/actualHits))
print('false negatives', falseNegatives, '({:2.1f}%)'.format(100*falseNegatives/actualHits))
print('')

print('true negatives', trueNegatives, '({:2.1f}%)'.format(100*trueNegatives/actualMisses))
print('false positives', falsePositives, '({:2.1f}%)'.format(100*falsePositives/actualMisses))

#plt.figure(2)
#plt.scatter(X[:,0],X[:,1],c=ypred,cmap=plt.cm.Spectral)

#plt.figure(3)
#plt.scatter(X[:,0],X[:,1],c=np.round(ypred),cmap=plt.cm.Spectral)

for i in range(2):
    plt.figure(i)
    plt.imshow(W1[i,:25].reshape((5,5)))
