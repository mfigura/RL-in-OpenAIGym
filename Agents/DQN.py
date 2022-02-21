import torch
import torch.nn.functional as F
from torch import nn,optim,distributions
from torch.utils.data import DataLoader, TensorDataset
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
        self.Q_optimizer = optim.SGD(self.Q_net.parameters(), lr=lrate)
        self.gamma = gamma
        self.n_actions = n_actions

    def _Q_update(self,s,a,r,ns,n_ep,n_refresh=1):
        '''
        Train the Q network to minimize the mean squared projected Bellman error
        Arguments: states, actions, rewards, new states, number of training epochs/fixed TD error, number of TD error updates
        '''
        with torch.no_grad():
            nQ = self.Q_net(ns).max(1).values
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
            print(f'Epoch: {i}, training loss: {train_loss}, validation loss: {valid_loss}')
        return train_loss

    def update(self,states,actions,rewards,new_states,n_epochs):
        '''Update Q-network'''
        critic_loss = self._Q_update(states,actions,rewards,new_states,n_epochs)
        return critic_loss

    def get_action(self,state,eps=0.1):
        '''Choose an optimal action with prob. 1-eps and random action with probability eps'''
        if np.random.binomial(1, eps) == 0:
            state = torch.tensor(state,dtype=torch.float32)
            with torch.no_grad():
                action = self.Q_net(state).argmax().numpy()
        else:
            action = np.random.choice(np.arange(self.n_actions))

        return action
