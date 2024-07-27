# import necessary modules

import torch
import torch.nn as nn

# Define simple small neural network model

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(8,4)
        self.fc2 = nn.Linear(4,1)

        # Define activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        # Pass the input through each layer
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x