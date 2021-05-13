# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal
"""

# define number of reporting macrocycles and epochs per reporting cycle
MacroCycles = 20
InnerCycles = 100

# optimization parameters
learning_rate = 0.02
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

fig,ax = plt.subplots(6)
sd  = model.state_dict()
for i in range(6):
    ax[i].cla()
    ax[i].imshow(sd['0.weight'][i].reshape([5,5]))
    
fig.canvas.draw_idle()
plt.pause(0.001)

for tt in range(MacroCycles):
    for t in range(InnerCycles):
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(tt*InnerCycles,loss.item())
    sd  = model.state_dict()
    for i in range(6):
        ax[i].cla()
        ax[i].imshow(sd['0.weight'][i].reshape([5,5]))
    
    fig.canvas.draw_idle()
    plt.pause(0.001)



  

