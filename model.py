import torch
import torch.functional as F
import torch.nn as nn
import random

class QNetwork():
    '''
    Creating a Feed Forward Linear Network for processing the states only.
    '''
    
    def __init__(self, state_size, action_size, f1_size, f2_size, seed=0):
        self.seed = random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, f1_size)
        self.fc2 = nn.Linear(f1_size, f2_size)
        self.fc3 = nn.Linear(f2_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x