import torch
import torch.nn.functional as F
from torch import nn,optim,distributions
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

'''
This is a realization of the deep deterministic policy gradient (DDPG) for single-agent RL environments.
The method is applicable in environments with continuous state spaces.
The code is designed for discrete action spaces.
'''

class DDPG_agent():

    '''
    ARGUMENTS: actor
               Q network
               actor learning rate
               Q learning rate
               number of possible actions
               discount factor gamma
    '''

    def __init__(self,actor,Q_net,actor_lrate,Q_lrate,n_actions,gamma=0.95):
        self.actor_model = actor
        self.actor_optimizer = optim.SGD(self.actor_model.parameters(), lr=actor_lrate)
        self.Q_net = Q_net
        self.Q_optimizer = optim.SGD(self.Q_net.parameters(), lr=Q_lrate)
        self.gamma = gamma
        self.n_actions = n_actions

    def _Q_update(self,s,a,r,ns,not_dones,n_ep,n_TD=10):
        '''
        Train the Q network to minimize the mean squared projected Bellman error
        Arguments: states, actions, rewards, new states, number of training epochs/fixed TD error, number of TD error updates
        '''
        not_dones = torch.Tensor(not_dones)
        for j in range(n_TD):
            with torch.no_grad():
                nQ = self.Q_net(ns).max(1).values * not_dones
                TD_targets = r + self.gamma*nQ.unsqueeze(-1)

            ds = TensorDataset(s,a,TD_targets)
            train_size = int(0.7 * len(ds))
            test_size = len(ds) - train_size
            train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
            s_train, a_train, y_train = train_ds[:]
            s_test, a_test, y_test = test_ds[:]

            a_train = a_train.to(torch.long).unsqueeze(-1)
            a_test = a_test.to(torch.long).unsqueeze(-1)
            for i in range(n_ep):
                self.Q_net.train()
                Q_train = self.Q_net(s_train).gather(1,a_train)
                train_loss = torch.square(y_train - Q_train).mean()
                self.Q_optimizer.zero_grad()
                train_loss.backward()
                self.Q_optimizer.step()
                self.Q_net.eval()
                with torch.no_grad():
                    Q_test = self.Q_net(s_test).gather(1,a_test)
                    valid_loss = torch.square(y_test - Q_test).mean()
            #print(f'Epoch: {j},{i}, training loss: {train_loss}, validation loss: {valid_loss}')
        return train_loss

    def _actor_update(self,s,a,r,ns,not_dones):
        '''
        Actor update
        Arguments: states, actions, rewards, new states
        '''
        not_dones = torch.Tensor(not_dones).unsqueeze(-1)
        act = a.to(torch.long).unsqueeze(-1)
        with torch.no_grad():
            nQ = self.Q_net(ns).max(1).values * not_dones
            Q = self.Q_net(s).gather(1,act)
            TD_errors = (r + self.gamma * nQ - Q).squeeze(-1)
        self.actor_model.train()
        a_prob = self.actor_model(s)
        loss = - (a_prob.log_prob(a) * TD_errors).sum()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss

    def update(self,states,actions,rewards,new_states,not_dones,n_epochs,n_TD):
        '''Update actor and critic networks'''
        critic_loss = self._Q_update(states,actions,rewards,new_states,not_dones,n_epochs,n_TD)
        actor_loss = self._actor_update(states,actions,rewards,new_states,not_dones)
        return critic_loss

    def get_action(self,state,eps=0.1):
        '''Choose an optimal action with prob. 1-eps and random action with probability eps'''
        if np.random.binomial(1, eps) == 0:
            state = torch.tensor(state,dtype=torch.float32)
            with torch.no_grad():
                action = self.actor_model(state).sample().numpy()
        else:
            action = np.random.choice(np.arange(self.n_actions))

        return action
