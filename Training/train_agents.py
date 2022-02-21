import numpy as np
import torch
import math
import gym
from gym import spaces
import pandas as pd

def train_batch(env,agent,args):
    '''
    ARGUMENTS: gym environment, RL agent, training parameters
    RETURNS: trained agent, simulation data
    '''
    paths = []
    gamma = args['gamma']
    k = args['update_frequency']
    n_episodes = args['n_episodes']
    max_ep_len = args['max_ep_len']
    n_epochs = args['n_epochs']

    observations, new_observations, actions, rewards = [], [], [], []
    ep_rewards = np.zeros(n_episodes)
    n_train_samples = 0

    for t in range(n_episodes):

        'EPISODE SIMULATION'

        new_obs = env.reset()
        #env.render()
        done = False

        counter = 0
        while not done:
            observations.append(new_obs)
            current_action = agent.get_action(observations[-1])
            new_obs, reward, done, _ = env.step(current_action)
            actions.append(current_action)
            new_observations.append(new_obs)
            rewards.append(reward)
            ep_rewards[t] += reward
            counter += 1            
            done = True if counter == max_ep_len else done

        n_train_samples += counter

        'SUMMARY OF THE TRAINING EPISODE'
        ep_rewards_avg = np.mean(ep_rewards[0:t+1]) if t < 100 else  np.mean(ep_rewards[t-99:t+1])
        print(f'| Episode: {t} | Rewards: {ep_rewards[t]} | Rewards cumulative average: {ep_rewards_avg}')
        path = {"Returns":ep_rewards}
        paths.append(path)

        'DATA PROCESSING AND ALGORITHM UPDATES'

        if n_train_samples >= k:

            sts = torch.tensor(np.array(observations),dtype=torch.float32)
            new_sts = torch.tensor(np.array(new_observations),dtype=torch.float32)
            acts = torch.tensor(np.array(actions),dtype=torch.float32)
            rews = torch.tensor(np.array(rewards),dtype=torch.float32).unsqueeze(-1)
            observations.clear(), new_observations.clear(), actions.clear(), rewards.clear()
            n_train_samples = 0

            critic_loss, actor_loss = agent.update(sts,acts,rews,new_sts,n_epochs)
            print(f'| Critic loss: {critic_loss} | Actor loss: {actor_loss}')

    sim_data = pd.DataFrame.from_dict(paths)
    return agent,sim_data
