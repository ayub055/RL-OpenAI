import gym
import d4rl
import pandas as pd
from typing import Tuple, List
from model import FNO1d
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import time
import wandb

class Data_AutoRegressive:
    def __init__(
            self: 'Data_AutoRegressive',
            observations: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            actions: np.ndarray,
            terminals: np.ndarray,
            history_len: int,
            k: int = 1,  # Number of future states to include
            device = torch.device('cpu')
    ) -> None:
        self.history_len = history_len
        self.future_reg_len = k
        self.next_observations = next_observations
        self._preproc_data(
            observations, next_observations, rewards, actions, terminals)
        self.device = device
    
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
                self.valid_indices = self.valid_indices[:-(self.future_reg_len-1)]
                c = 0
            c += 1
            i += 1
            
    def get_batch(
            self,
            batch_size: int,
            device: torch.device,
            idxs: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if idxs is None:
            batch_idxs = np.random.choice(self.valid_indices, size=batch_size, replace=False)
        else:
            batch_idxs = np.array(idxs)

        batch_observations = [
            [self.observations[idx - self.history_len + i] for i in range(self.history_len)]
            for idx in batch_idxs
        ]
        batch_next_observations = [
            [self.observations[idx + i] for i in range(0, self.future_reg_len)]
            for idx in batch_idxs
        ]
        batch_rewards = [self.rewards[idx - 1] for idx in batch_idxs]
        batch_actions = [[self.actions[idx - 1 + i] for i in range(0, self.future_reg_len)] for idx in batch_idxs]

        batch_observations = np.array(batch_observations).transpose(1, 0, 2)
        batch_next_observations = np.array(batch_next_observations).transpose(1, 0, 2)
        batch_rewards = np.array(batch_rewards).reshape(-1, 1)
        batch_actions = np.array(batch_actions).transpose(1, 0, 2)

        return (
            torch.from_numpy(batch_observations).to(device),
            torch.from_numpy(batch_actions).to(device),
            torch.from_numpy(batch_rewards).to(device),
            torch.from_numpy(batch_next_observations).to(device)
        )
    
            
    def add_trajectory(
        self,
        trajectory: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]], # s, a, r, s',
    ) -> None:
        observations, actions, rewards, next_observations = zip(*trajectory)
        self.observations.extend(observations)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.next_observations.extend(next_observations)

        # Update valid_indices based on the new data
        "TODO:"
        # raise NotImplementedError
    
    def add_trajectories(
        self,
        trajectories: List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]]
    ) -> None:
        for trajectory in trajectories:
            self.add_trajectory(trajectory)
            

class Data:
    def __init__(
            self: 'Data',
            observations: np.ndarray,
            next_observations: np.ndarray,
            rewards: np.ndarray,
            actions: np.ndarray,
            terminals: np.ndarray,
            history_len: int,
            device
    ) -> None:
        self.history_len = history_len
        self.next_observations = next_observations
        self._preproc_data(
            observations, next_observations, rewards, actions, terminals)
        self.device = device
    
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
    
    def get_batch(
            self,
            batch_size: int,
            device: torch.device,
            idxs: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if idxs is None:
            batch_idxs = np.random.choice(self.valid_indices, size=batch_size, replace=False)
        else:
            batch_idxs = np.array(idxs)
        batch_observations =\
            [[self.observations[idx-self.history_len+i] for i in range(self.history_len)] for idx in batch_idxs]
        batch_next_observation = [self.observations[idx] for idx in batch_idxs]
        batch_rewards = [self.rewards[idx-1] for idx in batch_idxs]
        batch_actions = [self.actions[idx-1] for idx in batch_idxs]
        
        batch_observations = np.array(batch_observations).transpose(0, 2, 1)
        batch_next_observation = np.array(batch_next_observation)[:, :, None]
        batch_rewards = np.array(batch_rewards).reshape(-1, 1)
        batch_actions = np.array(batch_actions)
        
        
        return ( # s, a, r, s'
            torch.from_numpy(batch_observations).to(device),
            torch.from_numpy(batch_actions).to(device),
            torch.from_numpy(batch_rewards).to(device),
            torch.from_numpy(batch_next_observation).to(device)
        )
    
    def add_trajectory(
        self,
        trajectory: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]], # s, a, r, s',
    ) -> None:
        observations, actions, rewards, next_observations = zip(*trajectory)
        self.observations.extend(observations)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.next_observations.extend(next_observations)

        # Update valid_indices based on the new data
        "TODO:"
        # raise NotImplementedError
    
    def add_trajectories(
        self,
        trajectories: List[List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]]
    ) -> None:
        for trajectory in trajectories:
            self.add_trajectory(trajectory)