import gym
import d4rl
import pandas as pd
from typing import Tuple
from model import FNO1d
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np



class ArgStorage:
    def __init__(self, args: dict) -> None:
        self.__dict__.update(args)


class Data:
    def __init__(
            self: 'Data',
            observations: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            actions: np.ndarray,
            terminals: np.ndarray,
            history_len: int
    ) -> None:
        self.history_len = history_len
        self.next_observations = next_observations
        self._preproc_data(
            observations, next_observations, rewards, actions, terminals)
    
    def _preproc_data(
            self,
            observations: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            actions: np.ndarray,
            terminals: np.ndarray,
    ) -> None:
        self.observations = []
        self.valid_indices = []
        self.actions = []
        self.rewards = []
        i, c = 0, 0
        while i < len(observations):
            self.observations.append(observations[i])
            self.actions.append(actions[i])
            self.rewards.append(rewards[i])
            if c >= self.history_len:
                self.valid_indices.append(i)
            if terminals[i]:
                self.observations.append(next_observations[i])
                self.rewards.append(0)
                self.actions.append(np.zeros_like(self.actions[-1]))
                c = 0
            c += 1
            i += 1
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
    
    def get_batch(
            self,
            batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_idxs = np.random.choice(self.valid_indices, size=batch_size, replace=False)
        # print(batch_idxs)
        batch_observations =\
            [self.observations[i-self.history_len:i] for i in batch_idxs]
            # [self.observations[i-self.history_len:i] for i in range(batch_idxs)]
        batch_observations = np.array(batch_observations)
        batch_next_observation = self.observations[batch_idxs]
        batch_rewards = self.rewards[batch_idxs-1]
        batch_actions = self.actions[batch_idxs-1]
        
        return ( # s, a, r, s'
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_next_observation,
        )
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.observations,
            self.actions,
            self.rewards,
            self.next_observations,
        )
    
    


def main():
    batch_size = 32  
    env = gym.make('halfcheetah-medium-v2')
    dataset = env.get_dataset()
    # print(dataset.keys())
    
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_observations = dataset['next_observations']
    terminals = dataset['terminals']
    timeouts = dataset['timeouts']
    terminals = np.logical_or(terminals, timeouts)
    
    data_loader = Data(observations, next_observations, rewards, actions, terminals, 5) 
    # TODO: vaiable history length

    train_observations, train_actions, train_rewards, train_next_observations = data_loader.get_data()
    train_size = int(len(train_observations) * 0.8)

    train_data = (
        train_observations[:train_size],
        train_actions[:train_size],
        train_rewards[:train_size],
        train_next_observations[:train_size]
    )

    test_data = (
        train_observations[train_size:],
        train_actions[train_size:],
        train_rewards[train_size:],
        train_next_observations[train_size:]
    )

    # print(train_data[0].shape)

    
    batch_observations, batch_actions, batch_rewards, batch_next_observations = data_loader.get_batch(batch_size)
    # print(batch_observations.shape)
    # print(batch_next_observations.shape)
    # print(batch_observations[0,0], batch_next_observations[0,0])


if __name__ == '__main__':
    main()
    
    # print(observations.shape, actions.shape, rewards.shape, terminals.shape, next_observations.shape)
    # print(observations[0], actions[0], rewards[0], terminals[0], next_observations[0])
    # print(observations[1], actions[1], rewards[1], terminals[1], next_observations[1])
    # print(observations[2], actions[2], rewards[2], terminals[2], next_observations[2])

    # print(observations[999], actions[999], rewards[999], terminals[999], next_observations[999])
    # print(observations[1000], actions[1000], rewards[1000], terminals[1000], next_observations[1000])
    # print(observations[1001], actions[1001], rewards[1001], terminals[1001], next_observations[1001])