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

    def _Q_update(self,train_set,test_set,n_ep,n_TD=10):
        '''
        Train the Q network to minimize the mean squared projected Bellman error
        Arguments: states, actions, rewards, new states, number of training epochs/fixed TD error, number of TD error updates
        '''
        s_train, a_train, r_train, ns_train, nd_train = train_set[:]
        s_test, a_test, r_test, ns_test, nd_test = test_set[:]
        act_train = a_train.to(torch.long)
        act_test = a_test.to(torch.long)

        for j in range(n_TD):
            with torch.no_grad():
                nQ_train = self.Q_net(ns_train).max(1,keepdim=True).values * nd_train.unsqueeze(-1)
                y_train = r_train + self.gamma * nQ_train
                nQ_test = self.Q_net(ns_test).max(1,keepdim=True).values * nd_test.unsqueeze(-1)
                y_test = r_test + self.gamma * nQ_test

            for i in range(n_ep):
                self.Q_net.train()
                Q_train = self.Q_net(s_train).gather(1,act_train)
                train_loss = ((y_train - Q_train)**2).mean()
                self.Q_optimizer.zero_grad()
                train_loss.backward()
                self.Q_optimizer.step()
                self.Q_net.eval()
                with torch.no_grad():
                    Q_test = self.Q_net(s_test).gather(1,act_test)
                    valid_loss = ((y_test - Q_test)**2).mean()
        return train_loss

    def _actor_update(self,train_set):
        '''
        Actor update
        Arguments: states, actions, rewards, new states
        '''
        s_train, a_train, r_train, ns_train, nd_train = train_set[:]
        act = a_train.to(torch.long)
        with torch.no_grad():
            Q_train = self.Q_net(s_train).gather(1,act)
            nQ_train = self.Q_net(ns_train).max(1,keepdim=True).values * nd_train.unsqueeze(-1)
            TD_errors = (r_train + self.gamma * nQ_train - Q_train).squeeze(-1)
        self.actor_model.train()
        log_policy_grad = - self.actor_model(s_train.squeeze()).log_prob(a_train.squeeze())
        loss = (log_policy_grad * TD_errors).sum()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        for param in self.actor_model.parameters():
            param.data = param.data.clamp(-10,10)
        return loss

    def update(self,s,a,r,ns,not_dones,n_epochs,n_TD):
        '''Update actor and critic networks'''
        ds = TensorDataset(s,a,r,ns,not_dones)
        train_size = int(0.7 * len(ds))
        test_size = len(ds) - train_size
        train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

        critic_loss = self._Q_update(train_ds,test_ds,n_epochs,n_TD)
        actor_loss = self._actor_update(train_ds)
        return critic_loss, actor_loss

    def get_action(self,state,eps=0.1):
        '''Choose an optimal action with prob. 1-eps and random action with probability eps'''
        if np.random.binomial(1, eps) == 0:
            state = torch.tensor(state,dtype=torch.float32)
            with torch.no_grad():
                action = self.actor_model(state).probs.argmax().numpy()
        else:
            action = np.random.choice(np.arange(self.n_actions))

        return action
