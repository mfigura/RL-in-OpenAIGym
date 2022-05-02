import numpy as np
import torch
import math
import gym
import random
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
    n_TD = args['n_TD']
    replay_buffer_size = args['replay_buffer_size']

    observations, new_observations, actions, rewards, not_dones = [], [], [], [], []
    ep_rewards = np.zeros(n_episodes)
    n_train_samples = 0

    for t in range(n_episodes):

        'EPISODE SIMULATION'

        new_obs = env.reset()
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
            not_dones.append(not done)
        n_train_samples += counter

        'SUMMARY OF THE TRAINING EPISODE'
        ep_rewards_avg = np.mean(ep_rewards[0:t+1]) if t < 100 else  np.mean(ep_rewards[t-99:t+1])
        print(f'| Episode: {t} | Rewards: {ep_rewards[t]} | Rewards cumulative average: {ep_rewards_avg}')
        path = {"Returns":ep_rewards}
        paths.append(path)

        'DATA PROCESSING AND ALGORITHM UPDATES'

        if n_train_samples >= k:
            'Process data'
            sts = torch.tensor(np.array(observations),dtype=torch.float32)
            new_sts = torch.tensor(np.array(new_observations),dtype=torch.float32)
            acts = torch.tensor(np.array(actions),dtype=torch.float32).unsqueeze(-1)
            rews = torch.tensor(np.array(rewards),dtype=torch.float32).unsqueeze(-1)
            not_dones = torch.Tensor(not_dones)
            'Update algorithm'
            critic_loss, actor_loss = agent.update(sts,acts,rews,new_sts,not_dones,n_epochs,n_TD)
            'Update replay buffers'
            n_removed_samples = max(n_train_samples - replay_buffer_size,0)
            if not n_removed_samples == 0:
                exp_replay_buffer = list(zip(observations,actions,rewards,new_observations,not_dones))
                random.shuffle(exp_replay_buffer)
                del exp_replay_buffer[:n_removed_samples]
                shuffled_zipped_ERB = list(zip(*exp_replay_buffer))
                observations = list(shuffled_zipped_ERB[0])
                actions = list(shuffled_zipped_ERB[1])
                rewards = list(shuffled_zipped_ERB[2])
                new_observations = list(shuffled_zipped_ERB[3])
                not_dones = list(shuffled_zipped_ERB[4])
                n_train_samples = n_train_samples - n_removed_samples
            'Print loss'
            print(f'| Critic loss: {critic_loss} | Actor loss: {actor_loss}')

    sim_data = pd.DataFrame.from_dict(paths)
    return agent,sim_data
