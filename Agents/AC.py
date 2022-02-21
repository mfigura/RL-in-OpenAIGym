import torch
import torch.nn.functional as F
from torch import nn,optim,distributions
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
               discount factor gamma
    '''

    def __init__(self,actor,critic,lrate,gamma=0.95):
        self.actor_model = actor
        self.actor_optimizer = optim.SGD(self.actor_model.parameters(), lr=lrate)
        self.critic_model = critic
        self.critic_optimizer = optim.SGD(self.critic_model.parameters(), lr=lrate)
        self.gamma = gamma

    def _critic_update(self,s,a,r,ns,n_ep):
        '''
        Trains a critic to minimize the mean squared projected Bellman error
        Arguments: states, actions, rewards, new states, number of training epochs
        '''
        for i in range(n_ep):
            with torch.no_grad():
                nV = self.critic_model(ns)
                TD_targets = r + self.gamma*nV
            self.critic_model.train()
            V = self.critic_model(s)
            loss = ((TD_targets - V)**2).mean()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            #print(i,loss)
        return loss

    def _actor_update(self,s,a,r,ns):
        '''
        Actor update
        Arguments: states, actions, rewards, new states
        '''
        with torch.no_grad():
            TD_errors = (r + self.gamma * self.critic_model(ns) - self.critic_model(s)).squeeze(-1)
        self.actor_model.train()
        a_prob = self.actor_model(s)
        loss = - (a_prob.log_prob(a) * TD_errors).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss

    def update(self,states,actions,rewards,new_states,n_epochs):
        '''Update actor and critic networks'''
        critic_loss = self._critic_update(states,actions,rewards,new_states,n_epochs)
        actor_loss = self._actor_update(states,actions,rewards,new_states)
        return critic_loss, actor_loss

    def get_action(self,state):
        '''Choose an action from the policy at the current state'''
        state = torch.tensor(state,dtype=torch.float32)
        a_prob = self.actor_model(state)
        action = a_prob.sample()
        return action.numpy()
