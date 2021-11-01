import torch
import torch.nn as nn
import torch.nn.functional as F
    
class neural_net(nn.Module):
    def __init__(self, num_sensors, params):
        super(neural_net,self).__init__()

        self.lstm = nn.LSTM(num_sensors,params[0])
        self.layer1 = nn.Linear(params[0], params[1])
        self.layer2 = nn.Linear(params[1],params[2])
        self.layer3 = nn.Linear(params[2],num_sensors)

    def forward(self,x):
        x = x.unsqueeze(dim=0)
        y,(_,_) = self.lstm(x)
        y = y.squeeze()
        y = self.layer1(y)
#         y = self.layer1(x)
        y_hat = torch.relu(y)
        y_hat = F.dropout(y_hat,0.2)
        z = self.layer2(y_hat)
        z_hat = torch.relu(z)
        z_hat = F.dropout(z_hat, 0.2)
        w = self.layer3(z_hat)
        return w
