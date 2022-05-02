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
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_lrate)
        self.critic_model = critic
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_lrate)
        self.gamma = gamma

    def _critic_update(self,train_set,test_set,n_ep,n_TD):
        '''
        Trains a critic to minimize the mean squared projected Bellman error
        Arguments: states, actions, rewards, new states, number of training epochs
        '''
        s_train, a_train, r_train, ns_train, nd_train = train_set[:]
        s_test, a_test, r_test, ns_test, nd_test = test_set[:]

        for j in range(n_TD):
            with torch.no_grad():
                nV_train = self.critic_model(ns_train) * nd_train.unsqueeze(-1)
                y_train = r_train + self.gamma * nV_train
                nV_test = self.critic_model(ns_test) * nd_test.unsqueeze(-1)
                y_test = r_test + self.gamma * nV_test

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
        return (train_loss,valid_loss)

    def _actor_update(self,train_set):
        '''
        Actor update
        Arguments: states, actions, rewards, new states
        '''
        s_train, a_train, r_train, ns_train, nd_train = train_set[:]
        with torch.no_grad():
            V_train = self.critic_model(s_train)
            nV_train = self.critic_model(ns_train) * nd_train.unsqueeze(-1)
            TD_errors = (r_train + self.gamma * nV_train - V_train).squeeze()
        self.actor_model.train()
        log_policy = - self.actor_model(s_train.squeeze()).log_prob(a_train.squeeze())
        train_loss = (log_policy * TD_errors).sum()
        self.actor_optimizer.zero_grad()
        train_loss.backward()
        self.actor_optimizer.step()
        for param in self.actor_model.parameters():
            param.data = param.data.clamp(-10,10)
        return train_loss

    def update(self,s,a,r,ns,not_dones,n_epochs,n_TD):
        '''Update actor and critic networks'''
        ds = TensorDataset(s,a,r,ns,not_dones)
        train_size = int(0.7 * len(ds))
        test_size = len(ds) - train_size
        train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

        critic_loss = self._critic_update(train_ds,test_ds,n_epochs,n_TD)
        actor_loss = self._actor_update(train_ds)
        return critic_loss, actor_loss

    def get_action(self,state):
        '''Choose an action from the policy at the current state'''
        state = torch.tensor(state,dtype=torch.float32)
        a_prob = self.actor_model(state)
        action = a_prob.sample()
        return action.numpy()
