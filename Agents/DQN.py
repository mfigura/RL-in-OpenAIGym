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

    def __init__(self,critic,lrate,gamma=0.95,action_space_dim=1):
        self.critic_model = critic
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=lrate)
        self.critic_loss = nn.MSELoss(reduction='mean')
        self.gamma = gamma
        self.action_space_dim = action_space_dim
        self.possible_actions = torch.Tensor(np.arange(action_space_dim))

    def get_optimal_action(self,state)
        sa = torch.cat((state.repeat(self.n_actions),self.possible_actions),axis=1)
        optimal_action = torch.argmax(self.critic_model(sa))

    def _critic_update(self,s,ns,r,n_ep):
        '''
        Trains a critic to minimize the mean squared projected Bellman error
        Arguments: state-action pairs, actions, new states, rewards, number of training epochs
        '''
        for i in range(n_ep):
            nQ = self.critic_model(nsa)
            TD_targets = r + self.gamma*nQ
            self.critic_model.train()
            Q = self.critic_model(sa)
            loss = self.critic_loss(Q,TD_targets)
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
        return loss

    def update(self,states,actions,new_states,rewards,n_epochs):
        '''Update Q-network'''
        critic_loss = self._critic_update(states,actions,new_states,rewards,n_epochs)
        return critic_loss

    def get_action(self,state,eps=0.1):
        '''Choose an optimal action with prob. 1-eps and random action with probability eps'''
        optimal_action = get_optimal_action(state).detach().numpy()
        random_action = np.random.choice(action_prob.shape[0], p = action_prob)

        return action, optimal_action
