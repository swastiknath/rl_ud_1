'''
AUTHOR: SWASTIK NATH.

'''
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = int(1e-3)
LR = 5e-4
UPDATE_EVERY=4

device = 'CUDA' if torch.cuda.is_available() else 'CPU'

class Agent():
    
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size=action_size
        self.seed = random.seed(seed)
        self.qnet_local = QNetwork(state_size, action_size, BATCH_SIZE, BATCH_SIZE, seed).to(device)
        self.qnet_target = QNetwork(self.state_size, self.action_size, BATCH_SIZE, BATCH_SIZE, seed).to(device)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=LR)
        self.memory = ExperienceBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE)
        self.time_step = 0
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.time_step = (self.time_step + 1) % UPDATE_EVERY:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        qtargets_next = self.qnet_target(next_states).detach().max(1).unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.qnet_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_local)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def act(self, state, epsilon = 0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()
        if random.random > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random_choice(np.arange(self.action_size))
        
    def soft_update(self, local_mode, target_mode, tau):
        for target_param, local_param in zip(target_mode, local_mode):
            target_param.data_copy(tau * local_param.data + (1 - tau) * target_param.data) 
        
class ExperienceBuffer():
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = random.seed(seed)
        self.action_size= action_size
        self.experience = namedTuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action , reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e in not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)
    
    def memStat(self):
        return len(self.memory)