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
               learning rate
               number of possible actions
               discount factor gamma
    '''

    def __init__(self,Q_net,lrate,n_actions,gamma=0.95):
        self.Q_net = Q_net
        self.Q_optimizer = optim.Adam(self.Q_net.parameters(), lr=lrate)
        self.gamma = gamma
        self.n_actions = n_actions

    def _Q_update(self,s,a,r,ns,not_dones,n_ep,n_TD=10):
        '''
        Train the Q network to minimize the mean squared projected Bellman error
        Arguments: states, actions, rewards, new states, number of training epochs/fixed TD error, number of TD error updates
        '''
        ds = TensorDataset(s,a,r,ns,not_dones)
        train_size = int(0.7 * len(ds))
        test_size = len(ds) - train_size
        train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
        s_train, a_train, r_train, ns_train, nd_train = train_ds[:]
        s_test, a_test, r_test, ns_test, nd_test = test_ds[:]
        a_train = a_train.to(torch.long)
        a_test = a_test.to(torch.long)

        for j in range(n_TD):
            '''Evaluate TD targets'''
            with torch.no_grad():
                nQ_train = self.Q_net(ns_train).max(1).values * nd_train
                y_train = r_train + self.gamma*nQ_train.unsqueeze(-1)
                nQ_test = self.Q_net(ns_test).max(1).values * nd_test
                y_test = r_test + self.gamma*nQ_test.unsqueeze(-1)
            '''Minimize mean squared error'''
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

        return (train_loss,valid_loss)

    def update(self,states,actions,rewards,new_states,not_dones,n_epochs,n_TD):
        '''Update Q-network'''
        critic_loss = self._Q_update(states,actions,rewards,new_states,not_dones,n_epochs,n_TD)
        return critic_loss, None

    def get_action(self,state,eps=0.1):
        '''Choose an optimal action with prob. 1-eps and random action with probability eps'''
        if np.random.binomial(1, eps) == 0:
            state = torch.tensor(state,dtype=torch.float32)
            with torch.no_grad():
                action = self.Q_net(state).argmax().numpy()
        else:
            action = np.random.choice(np.arange(self.n_actions))

        return action
