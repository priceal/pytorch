# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal

"""

# define number of reporting macrocycles and epochs per reporting cycle
MacroCycles = 50
InnerCycles = 100

# layer to plot
HL = H
weights = '0.weight'

# optimization parameters
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

fig,ax = plt.subplots(HL)
sd  = model.state_dict()
for i in range(HL):
    ax[i].cla()
    ax[i].imshow(sd[weights][i].reshape([5,5]))
    
fig.canvas.draw_idle()
plt.pause(0.001)

loss_trajectory = []
for tt in range(MacroCycles):
    for t in range(InnerCycles):
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    yt_pred = model(xt)    
    loss_test = loss_fn(yt_pred,yt)
    print(tt*InnerCycles,loss.item(),loss_test.item())
    loss_trajectory.append([loss.item(),loss_test.item()])
    sd  = model.state_dict()
    for i in range(HL):
        ax[i].cla()
        ax[i].imshow(sd[weights][i].reshape([5,5]))
    
    fig.canvas.draw_idle()
    plt.pause(0.001)
loss_array = np.array(loss_trajectory)
plt.figure()
plt.plot(loss_array[:,0])
plt.plot(loss_array[:,1])

  

