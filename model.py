'''
AUTHOR : SWASTIK NATH
UDACITY DEEP REINFORCEMENT LEARNING NANODEGREE
DQN Q NETWORK 
IMPLEMENTATION IN COURTESY OF UDACITY.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNetwork(nn.Module):
    '''
    QNETWORK TO MAP STATES TO ACTIONS WITH FEEDFORWARD AND DUELING. 
    '''
    def __init__(self, state_size, action_size, seed=5, use_dueling=False, fc1_size=64, fc2_size=64):
        '''
        PARAM:
        STATE_SIZE: SIZE OF THE STATE SPACE OF THE ENVIRONMENT
        ACTION_SIZE:SIZE OF THE ACTION SPACE OF THE ENVIRONMENT
        SEED: RANDOM SEED 
        USE_DUELING: DUELING ARCHITECTURE OF THE DQN
        FC1_SIZE: SIZE OF FC LAYER 1
        FC2_SIZE: SIZE OF FC LAYER 2
        '''
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = use_dueling
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)
        self.a_state = nn.Linear(fc2_size, 1)
    def forward(self, state):
        '''
        PARAM:
        STATE: STATE VALUES IN DEVICE SPECIFIC TORCH TENSOR
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.dueling:
            return self.fc3(x) + self.a_state(x)
        else:
            return self.fc3(x)        