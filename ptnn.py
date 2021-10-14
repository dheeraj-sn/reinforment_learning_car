"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Activation, Dropout
#from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.layers.recurrent import LSTM
#from tensorflow.keras.callbacks import Callback

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
"""

# def neural_net(num_sensors, params):
    
class neural_net(nn.Module):
    def __init__(self, num_sensors, params):
        super(neural_net,self).__init__()

        self.layer1 = nn.Linear(num_sensors, params[0])
        self.layer2 = nn.Linear(params[0],params[1])
        self.layer3 = nn.Linear(params[1],num_sensors)

    def forward(self,x):
        y = self.layer1(x)
        y_hat = torch.relu(y)
        y_hat = F.dropout(y_hat,0.2)
        z = self.layer2(y_hat)
        z_hat = torch.relu(z)
        z_hat = F.dropout(z_hat, 0.2)
        w = self.layer3(z_hat)
        return w

    

#     if load:
#         model.load_weights(load)

#     return model
