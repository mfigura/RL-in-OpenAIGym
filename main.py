import numpy as np
import torch
from torch import nn, optim, distributions
import torch.nn.functional as F
import gym
import argparse
import math
from gym import spaces
from Agents.AC import AC_agent
from Agents.DQN import DQN_agent
from Agents.DDPG import DDPG_agent
from Training.train_agents import train_batch

'''
This is a main file, where the user selects a gym environment, training and simulation parameters,
and neural network architecture for the approximation of the policy and value function. The script triggers
a training process whose results are passed to folder Simulation_results.
'''

class Value_Approximation(nn.Module):
    '''Neural network V(s;w,b)'''
    def __init__(self,obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,20)
        self.fc4 = nn.Linear(20,1)
        self.relu_layer = nn.LeakyReLU(0.1)

    def forward(self, x):
        z1 = self.relu_layer(self.fc1(x))
        z2 = self.relu_layer(self.fc2(z1))
        z3 = self.relu_layer(self.fc3(z2))
        return self.fc4(z3)

class Q_Approximation(nn.Module):
    '''Neural network Q(s;w,b)'''
    def __init__(self,obs_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,20)
        self.fc4 = nn.Linear(20,action_dim)
        self.relu_layer = nn.LeakyReLU(0.1)

    def forward(self, x):
        z1 = self.relu_layer(self.fc1(x))
        z2 = self.relu_layer(self.fc2(z1))
        z3 = self.relu_layer(self.fc3(z2))
        return self.fc4(z2)

class Policy_Approximation(nn.Module):
    '''Neural network pi(a|s;w,b)'''
    def __init__(self,obs_dim,action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim,20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,action_dim)
        self.softmax_layer = nn.Softmax(-1)
        self.relu_layer = nn.LeakyReLU(0.3)

    def forward(self, x):
        z1 = self.relu_layer(self.fc1(x))
        z2 = self.relu_layer(self.fc2(z1))
        z3 = self.softmax_layer(self.fc3(z2))
        return distributions.Categorical(z3)

if __name__ == '__main__':

    '''USER-DEFINED PARAMETERS'''
    parser = argparse.ArgumentParser(description='Provide parameters for training RL agents')
    parser.add_argument('--n_episodes', help='number of episodes', type=int, default=2000)
    parser.add_argument('--max_ep_len', help='max episode length', type=int, default=200)
    parser.add_argument('--update_frequency', help='min number of samples for an update', type=int, default=2000)
    parser.add_argument('--critic_lr', help='value approximation learning rate',type=float, default=0.002)
    parser.add_argument('--actor_lr', help='policy approximation learning rate (if applicable)',type=float, default=0.005)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.9)
    parser.add_argument('--n_epochs',help='number of training epochs for the critic',type=int,default=10)
    parser.add_argument('--n_TD',help='number of times TD errors are reevaluated in the critic updates',type=int,default=10)
    parser.add_argument('--replay_buffer_size',help='number of samples that are stored in the replay buffer',type=int,default=1000)
    parser.add_argument('--random_seed',help='Set random seed for the random number generator',type=int,default=10)

    args = vars(parser.parse_args())
    np.random.seed(args['random_seed'])
    print(args)
    #----------------------------------------------------------------------------------------------------------------------------------------
    '''LOAD GYM ENVIRONMENT'''
    env = gym.make('CartPole-v1')
    n_actions = env.action_space.n
    obs_dim = env.observation_space.sample().shape[0]
    print(n_actions,obs_dim)
    #----------------------------------------------------------------------------------------------------------------------------------------
    '''CREATE AGENT'''
    agent = AC_agent(actor=Policy_Approximation(obs_dim,n_actions),critic=Value_Approximation(obs_dim),actor_lrate=args['actor_lr'],critic_lrate=args['critic_lr'],gamma=args['gamma'])
    #agent = DQN_agent(Q_net=Q_Approximation(obs_dim,n_actions),lrate=args['critic_lr'],n_actions=n_actions,gamma=args['gamma'])
    #agent = DDPG_agent(actor=Policy_Approximation(obs_dim,n_actions),Q_net=Q_Approximation(obs_dim,n_actions),actor_lrate=args['actor_lr'],Q_lrate=args['critic_lr'],n_actions=n_actions,gamma=args['gamma'])
    #---------------------------------------------------------------------------------------------------------------------------------------------
    '''TRAIN AGENTS'''
    trained_agents,sim_data = train_batch(env,agent,args)
    sim_data.to_pickle("./Simulation_results/sim_data.pkl")
