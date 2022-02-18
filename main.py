import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import gym
import argparse
import math
from gym import spaces
from Agents.AC import AC_agent
from Training.train_agents import train_batch

'''
This is a main file, where the user selects a gym environment, training and simulation parameters,
and neural network architecture for the approximation of the policy and value function. The script triggers
a training process whose results are passed to folder Simulation_results.
'''

class ValueApproximation(nn.Module):
    '''Neural network V(s;w,b)'''
    def __init__(self,in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(self.in_dim,2*self.in_dim)
        self.fc2 = nn.Linear(2*self.in_dim,2*self.in_dim)
        self.fc3 = nn.Linear(2*self.in_dim, 1)
        self.relu_layer = nn.LeakyReLU(0.3)

    def forward(self, x):
        z1 = self.relu_layer(self.fc1(x))
        z2 = self.relu_layer(self.fc2(z1))
        return self.fc3(z2)

class PolicyApproximation(nn.Module):
    '''Neural network pi(a|s;w,b)'''
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim,2*self.in_dim)
        self.fc2 = nn.Linear(2*self.in_dim,2*self.in_dim)
        self.fc3 = nn.Linear(2*self.in_dim,self.out_dim)
        self.softmax_layer = nn.Softmax(-1)
        self.relu_layer = nn.LeakyReLU(0.3)

    def forward(self, x):
        z1 = self.relu_layer(self.fc1(x))
        z2 = self.relu_layer(self.fc2(z1))
        return self.softmax_layer(self.fc3(z2))

if __name__ == '__main__':

    '''USER-DEFINED PARAMETERS'''
    parser = argparse.ArgumentParser(description='Provide parameters for training RL agents')
    parser.add_argument('--n_episodes', help='number of episodes', type=int, default=1000)
    parser.add_argument('--max_ep_len', help='max episode length', type=int, default=200)
    parser.add_argument('--update_frequency', help='Frequency of updates (episodes/update)', type=int, default=50)
    parser.add_argument('--lr', help='function approximation learning rate',type=float, default=0.01)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.9)
    parser.add_argument('--n_epochs',help='number of training epochs for the critic',type=int,default=10)
    parser.add_argument('--random_seed',help='Set random seed for the random number generator',type=int,default=None)

    args = vars(parser.parse_args())
    np.random.seed(args['random_seed'])
    print(args)
    #----------------------------------------------------------------------------------------------------------------------------------------
    '''LOAD GYM ENVIRONMENT'''
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    obs_dim = env.observation_space.sample().shape[0]
    #----------------------------------------------------------------------------------------------------------------------------------------
    '''CREATE AGENT'''
    actor = PolicyApproximation(obs_dim,n_actions)
    critic = ValueApproximation(obs_dim)
    agent = AC_agent(actor,critic,args['lr'],args['gamma'])
    #---------------------------------------------------------------------------------------------------------------------------------------------
    '''TRAIN AGENTS'''
    trained_agents,sim_data = train_batch(env,agent,args)
    sim_data.to_pickle("./Simulation_results/sim_data.pkl")
