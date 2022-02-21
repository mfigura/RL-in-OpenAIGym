import torch
import torch.nn.functional as F
from torch import nn,optim
import numpy as np
import math

'''
This is a realization of deep Q-learning (DQN) for single-agent RL environments.
The method is applicable for environments with continuous state spaces.
The code is designed for discrete action spaces.
'''

class DQN_agent():
    '''
    ARGUMENTS: critic
               discount factor gamma
    '''

    def __init__(self,Q_net,lrate,n_actions,gamma=0.95):
        self.Q_net = Q_net
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=lrate)
        self.gamma = gamma
        self.n_actions = n_actions

    def _Q_update(self,s,a,r,ns,n_ep):
        '''
        Train the Q network to minimize the mean squared projected Bellman error
        Arguments: state-action pairs, actions, new states, rewards, number of training epochs
        '''
        for i in range(n_ep):
            with torch.no_grad():
                nQ = self.Q_net(ns).max(1).values
                TD_targets = r + self.gamma*nQ.unsqueeze(-1)
            self.Q_net.train()
            Q = self.Q_net(s)[:,a]
            loss = ((TD_targets - Q) ** 2).mean()
            self.Q_net.zero_grad()
            loss.backward()
            self.Q_net.step()
        return loss

    def update(self,states,actions,rewards,new_states,n_epochs):
        '''Update Q-network'''
        critic_loss = self._Q_update(states,actions,rewards,new_states,n_epochs)
        return critic_loss

    def get_action(self,state,eps=0.1):
        '''Choose an optimal action with prob. 1-eps and random action with probability eps'''
        if np.random.binomial(1, eps) == 0:
            state = torch.tensor(state,dtype=torch.float32)
            action = self.Q_net(state).argmax().numpy()
        else:
            action = np.random.choice(np.arange(self.n_actions))

        return action
