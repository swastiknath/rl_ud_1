'''
AUTHOR : SWASTIK NATH
UDACITY DEEP REINFORCEMENT LEARNING NANODEGREE
DQN RL AGENT
IMPLEMENTATION IN COURTESY OF UDACITY.
'''
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-4
TAU = 1e-3
UPDATE_EVERY = 4   #FREQUENCY OF LEARNING FROM EXPERIENCE REPLAY BUFFER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ExperienceReplayBuffer():
    '''
    SERVING AS THE EXPERIENCE REPLAY BUFFER OF THE DQN FRAMEWORK
    '''
    def __init__(self, action_size, buffer_size, batch_size, seed=0):
        '''
        PARAMS:
        ACTION_SIZE: SIZE OF THE ACTION SPACE OF THE ENVIRONMENT
        BUFFER_SIZE: SIZE OF THE EXPERIENCE REPLAY BUFFER
        BATCH_SIZE: SIZE OF THE BATCH 
        SEED : RANDOM SEED
        '''
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=self.buffer_size)
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'next_state', 'reward', 'done'])
        self.seed = random.seed(seed)
        
    def add(self, state, action, next_state, reward, done):
        '''
        ADDING THE S, A, S', R, DONE TUPLE TO THE EXPERIENCE
        PARAMS:
        STATE: STATE OF THE ENVIRONMENT
        ACTION: ACTION TAKEN BASED ON THE POLICY
        NEXT_STATE: NEXT STATE FROM THE ACTION
        REWARD: REWARD PROVIDED FROM THE ENVIRONMENT
        DONE: TERMINATION SIGNAL OF THE EPISODE
        '''
        e = self.experience(state, action, next_state, reward, done)
        self.memory.append(e)
        
    def sample(self):
        '''
        RETURNING SAMPLES OF THE S, A, S', R, DONE TUPLES FROM THE MEMORY 
        TO LEARN FROM THEM LATER.
        '''
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, next_states, rewards, dones)

        
    def __len__(self):
        '''
        RETURNING LENGTH OF THE MEMORY.
        '''
        return len(self.memory)

class Agent():
    
    def __init__(self, state_size, action_size, seed=0, use_dueling=False, use_double_dqn=False):
        '''
        PARAMS:
        STATE_SIZE: SIZE OF THE STATES OF THE ENVIRONMENT.
        ACTION_SIZE: SIZE OF THE ACTION SPACE OF THE ENVIRONMENT.
        SEED: RANDOM SEED.
        USE_DUELING: DQN DUELING ARCHITECTURE
        USE_DOUBLE_DQN: USING THE DOUBLE DQN ARCHITECTURE
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_dueling = use_dueling
        self.use_double_dqn = use_double_dqn
        
        self.qnetwork_local = QNetwork(state_size, action_size, seed, use_dueling=use_dueling)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, use_dueling=use_dueling)
        print(f"Local Q Network Configuration: {self.qnetwork_local}")
        print(f"Target Q Network Configuration: {self.qnetwork_target}")
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ExperienceReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        '''
        ADDING S, A, R, S' DONE TUPLES TO THE EXPERIENCE BUFFER AND LEARNING FROM THEM.
        PARAMS:
        STATE: STATE OF THE ENVIRONMENT
        ACTION: ACTION TAKEN BASED ON THE POLICY
        NEXT_STATE: NEXT STATE FROM THE ACTION
        REWARD: REWARD PROVIDED FROM THE ENVIRONMENT
        DONE: TERMINATION SIGNAL OF THE EPISODE
        '''
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                
    def act(self, state, eps=0.):
        '''
        CHOOSING ACTION FOLLOWING THE EPSILON GREEDY POLICY:
        PARAMS:
        STATE:STATE OF THE ENVIRONMENT
        EPS: EPSILON IMPLEMENTED WITH DECAY OVER TIMESTEPS.
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_val = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_val.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experience, gamma):
        '''
        LEARNING FROM EXPERIENCE REPLAY BUFFER WITH SOFT UPDATES OVER 
        THE LOCAL AND TARGET QNETWORK. 
        EXPERIENCE: S, A, R, S' DONE TUPLES SAMPLED FROM THE EXPERIENCE REPLAY BUFFER MEMORY.
        GAMMA: HYPERPARAM FOR THE DISCOUNT FACTOR.
        '''
        states, actions, rewards,next_states, dones = experience
        if self.use_double_dqn:
            indices = torch.argmax(self.qnetwork_local(next_states).detach(),1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,indices.unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        '''
        APPLYING SOFT-UPDATE OVER THE TARGET QNETWORK:
        THETA' = TAU * THETA + (1-TAU)*THETA'
        PARAMS:
        LOCAL_MODEL: LOCAL QNETWORK
        TARGET_MODEL: TARGET QNETWORK
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau) * target_param.data)
            