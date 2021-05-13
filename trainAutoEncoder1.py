# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal
"""

# define number of reporting macrocycles and epochs per reporting cycle
MacroCycles = 1
InnerCycles = 1

# optimization parameters
learning_rate = 0.02
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

fig,ax = plt.subplots(1)
ax.set_xlim([xMin, xMax])
ax.set_ylim([yMin, yMax])
y_pred = model(x_all).detach().numpy()
ax.scatter(y_pred[:,0],y_pred[:,1])
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
    y_pred = model(x_all).detach().numpy()
    ax.cla()
    ax.set_xlim([xMin, xMax])
    ax.set_ylim([yMin, yMax])
    ax.scatter(y_pred[:,0],y_pred[:,1])
    fig.canvas.draw_idle()
    plt.pause(0.001)



  

