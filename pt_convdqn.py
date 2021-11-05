import torch
import torch.nn as nn
import torch.nn.functional as F
    
class neural_net(nn.Module):
    def __init__(self, num_sensors, params):
        super(neural_net,self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=1)
        self.layer1 = nn.Linear(192, params[0])
        self.layer2 = nn.Linear(params[0],params[1])
        self.layer3 = nn.Linear(params[1],num_sensors)

    def forward(self,x):
        x = x.unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        y = self.conv1(x)
        y = torch.relu(y)
        y = self.conv2(y)
        y = torch.relu(y)
        y = y.squeeze(dim=-2)
        y = y.view(y.size(0),-1)
        y = self.layer1(y)
        y_hat = torch.relu(y)
        y_hat = F.dropout(y_hat,0.2)
        z = self.layer2(y_hat)
        z_hat = torch.relu(z)
        z_hat = F.dropout(z_hat, 0.2)
        w = self.layer3(z_hat)
        return w