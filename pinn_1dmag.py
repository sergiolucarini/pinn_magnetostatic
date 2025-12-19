
"""
Created on Wed Aug 27 16:08:46 2025

@authors: Sergio Lucarini
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

## NN
class Net(torch.nn.Module):
    # layers
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(1,10)
        self.hidden_layer2 = torch.nn.Linear(10,10)
        self.hidden_layer3 = torch.nn.Linear(10,10)
        self.hidden_layer4 = torch.nn.Linear(10,10)
        self.hidden_layer5 = torch.nn.Linear(10,10)
        self.output_layer = torch.nn.Linear(10,1)
    # activations
    def forward(self, x):
        inputs = torch.cat([x],axis=1) # combined
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## For regression, no activation is used in output layer
        return output

## Model
net = Net()
device = torch.device("cpu")
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters())# Adaptive Moment Estimation

## PDE 
def f(x, net):
    mu_0=4*np.pi*1e-1 # vacuum magnetic permeability
    u = net(x) # the dependent variable u is given by the network based on independent variable x
    u1=torch.reshape(u[:,0],[-1,1]);

    # derivatives
    u1_x = torch.autograd.grad(u1.sum(), x, create_graph=True)[0]
    muu1_x_x = torch.autograd.grad(mu_0*u1_x.sum(), x, create_graph=True)[0]
    
    # pde
    pde = muu1_x_x
    return pde

## Training
epochss = 5000;
loss_list=[]
for epoch in range(epochss):
    optimizer.zero_grad() # to make the gradients zero

    # Loss based on BCs
    x_bc = np.zeros([200,1])
    x_bc[200//2:,0]=1.0 # for x=0 / x=1
    u_bc = np.zeros((200,1))
    u_bc[200//2:,0]=1.0 # u=0 / u=1
    pt_x_bc = torch.autograd.Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    pt_u_bc = torch.autograd.Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    net_bc_out = net(pt_x_bc) # output of u=nn(x)
    mse_bc = mse_cost_function(net_bc_out, pt_u_bc)
    
    # Loss based on PDE
    x_collocation = np.random.uniform(low=0.0, high=1.0, size=(2000,1))
    all_zeros = np.zeros((2000,1))
    pt_x_collocation = torch.autograd.Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = torch.autograd.Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    f_out = 1e-2*f(pt_x_collocation, net) # output of pde(x)
    mse_f = mse_cost_function(f_out, pt_all_zeros)

    # Combining the loss functions and minimizing
    loss = mse_bc + mse_f
    loss.backward() # computing gradients using backward propagation
    optimizer.step() # update nn parameters
    
    # output for monitoring
    with torch.autograd.no_grad():
        if epoch%100==0: print(epoch,"Traning Loss:",loss.data);loss_list.append(loss.data)
    if epoch%100==1:
        fig = plt.figure()
        x=np.arange(0,1,0.02)
        ms_x=x
        x = np.ravel(np.arange(0,1,0.02)).reshape(-1,1)
        pt_x = torch.autograd.Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
        pt_u = net(pt_x)
        u=pt_u.data.cpu().numpy()
        ms_u = u[:,0].reshape(ms_x.shape)
        if epoch==0 and (np.any(ms_u<0) or np.any(ms_u>1)): break
        plt.plot(ms_x,ms_u)
        plt.show()
        plt.close()
fig = plt.figure()
plt.plot(loss_list)
plt.show()
plt.close()
