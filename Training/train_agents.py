import numpy as np
import torch
import math
import gym
from gym import spaces
import pandas as pd

def list2tensor(obs,new_obs,acts,rews):
    'Converts training data from lists to tensors'
    states = torch.tensor(np.array(obs),dtype=torch.float32)
    new_states = torch.tensor(np.array(new_obs),dtype=torch.float32)
    actions = torch.tensor(np.array(acts),dtype=torch.float32)
    rewards = torch.tensor(np.array(rews),dtype=torch.float32).unsqueeze(-1)


    return states,new_states,actions,rewards

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

    for t in range(n_episodes):

        'EPISODE SIMULATION'

        initial_state = env.reset()
        #env.render()
        done = False
        observations.append(initial_state)

        counter = 0
        while not done:
            current_state = observations[-1]
            current_action = agent.get_action(current_state)
            new_state, reward, done, _ = env.step(current_action)
            actions.append(current_action)
            new_observations.append(new_state)
            rewards.append(reward)
            ep_rewards[t] += reward
            done = True if counter == max_ep_len else done
            if not done:
                observations.append(new_state)
            counter += 1

        'SUMMARY OF THE TRAINING EPISODE'
        ep_rewards_avg = np.mean(ep_rewards[0:t+1]) if t < 100 else  np.mean(ep_rewards[t-99:t+1])
        print(f'| Episode: {t} | Rewards: {ep_rewards[t]} | Rewards cumulative average: {ep_rewards_avg}')
        path = {"Returns":ep_rewards}
        paths.append(path)

        'DATA PROCESSING AND ALGORITHM UPDATES'

        if ((t+1) % k) == 0:
            print(len(observations))
            sts, new_sts, acts, rews = list2tensor(observations,new_observations,actions,rewards)
            critic_loss, actor_loss = agent.update(sts,new_sts,rews,acts,n_epochs)
            print(f'| Critic loss: {critic_loss} | Actor loss: {actor_loss}')
            observations.clear(),new_observations.clear(),actions.clear(),rewards.clear()
    sim_data = pd.DataFrame.from_dict(paths)
    return agent,sim_data
