import torch
import torch.nn.functional as F
from torch import nn,optim,distributions
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

'''
This is a realization of the advantage actor-critic method (A2C) for single-agent RL environments.
The method is applicable in environments with continuous state spaces.
The code is designed for discrete action spaces.
'''

class AC_agent():

    '''
    ARGUMENTS: actor
               critic
               actor learning rate
               critic learning rate
               discount factor gamma
    '''

    def __init__(self,actor,critic,actor_lrate,critic_lrate,gamma=0.95):
        self.actor_model = actor
        self.actor_optimizer = optim.SGD(self.actor_model.parameters(), lr=actor_lrate)
        self.critic_model = critic
        self.critic_optimizer = optim.SGD(self.critic_model.parameters(), lr=critic_lrate)
        self.gamma = gamma

    def _critic_update(self,s,a,r,ns,not_dones,n_ep,n_TD):
        '''
        Trains a critic to minimize the mean squared projected Bellman error
        Arguments: states, actions, rewards, new states, number of training epochs
        '''
        not_dones = torch.Tensor(not_dones).unsqueeze(-1)
        for j in range(n_TD):
            with torch.no_grad():
                nV = self.critic_model(ns) * not_dones
                TD_targets = r + self.gamma * nV

            ds = TensorDataset(s,TD_targets)
            train_size = int(0.7 * len(ds))
            test_size = len(ds) - train_size
            train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
            s_train, y_train = train_ds[:]
            s_test, y_test = test_ds[:]

            for i in range(n_ep):
                self.critic_model.train()
                V_train = self.critic_model(s_train)
                train_loss = ((y_train - V_train)**2).mean()
                self.critic_optimizer.zero_grad()
                train_loss.backward()
                self.critic_optimizer.step()
                self.critic_model.eval()
                with torch.no_grad():
                    V_test = self.critic_model(s_test)
                    valid_loss = ((y_test - V_test)**2).mean()
            #print(f'Epoch: {i}, training loss: {train_loss}, validation loss: {valid_loss}')
        return train_loss

    def _actor_update(self,s,a,r,ns,not_dones):
        '''
        Actor update
        Arguments: states, actions, rewards, new states
        '''
        not_dones = torch.Tensor(not_dones).unsqueeze(-1)
        with torch.no_grad():
            TD_errors = (r + self.gamma * self.critic_model(ns) * not_dones - self.critic_model(s)).squeeze(-1)
        self.actor_model.train()
        a_prob = self.actor_model(s)
        loss = - (a_prob.log_prob(a) * TD_errors).sum()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss

    def update(self,states,actions,rewards,new_states,not_dones,n_epochs,n_TD):
        '''Update actor and critic networks'''
        critic_loss = self._critic_update(states,actions,rewards,new_states,not_dones,n_epochs,n_TD)
        actor_loss = self._actor_update(states,actions,rewards,new_states,not_dones)
        return critic_loss

    def get_action(self,state):
        '''Choose an action from the policy at the current state'''
        state = torch.tensor(state,dtype=torch.float32)
        a_prob = self.actor_model(state)
        action = a_prob.sample()
        return action.numpy()
