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
    
    def get_batch(
            self,
            batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_idxs = np.random.choice(self.valid_indices, size=batch_size, replace=False)
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
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_next_observation,
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


class SpectralConv1d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d_fast, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] =\
            self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class ConvBlock(nn.Module):
    def __init__(self, conv_channels, state_dim, ac_dim):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(conv_channels, conv_channels, 1)
        self.bn = torch.nn.BatchNorm1d(conv_channels)  # TODO: check is this is even required
        self.ac_dim = ac_dim
        self.ac_encoder = nn.Linear(ac_dim, conv_channels * state_dim)  # TODO: make it general (state dim independent)
    
    def forward(self, x, a):
        _, channels_, dim_ = x.shape
        x = self.conv(x) # 32 x 20 x 17
        a = self.ac_encoder(a).reshape(-1, channels_, dim_) # 32 x 20 x 17
        x = x + a
        x = self.bn(x) # 32 x 20 x 17


class FNO1d(nn.Module):
    def __init__(self, modes, width, history, state_dim, ac_dim):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 1 location (u(t-10, x), ..., u(t-1, x),  x)
        input shape: (batchsize, x=64, c=11)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, c=1)
        """

        self.modes = modes
        self.width = width
        self.history = history
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(self.history + 1, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.w0 = ConvBlock(width, state_dim, ac_dim)
        self.w1 = ConvBlock(width, state_dim, ac_dim)
        self.w2 = ConvBlock(width, state_dim, ac_dim)
        self.w3 = ConvBlock(width, state_dim, ac_dim)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, a): # x is state, a is action
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # 32 x 17 x (10+1)
        x = self.fc0(x)  # 32 x 17 x 20
        x = x.permute(0, 2, 1)  # 32 x 20 x 17
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)  # 32 x 20 x 17
        x2 = self.w0(x, a)  # 32 x 20 x 17
        x = x1 + x2
        x = F.gelu(x)  # 32 x 20 x 17

        x1 = self.conv1(x)
        x2 = self.w1(x, a)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x, a)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x, a)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batch_size, size_x = shape[0], shape[1]
        grid_x = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid_x = grid_x.reshape(1, size_x, 1).repeat([batch_size, 1, 1])
        return grid_x.to(device)


def main():
    batch_size = args.batch_size
    hist_len = args.history
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    n_iters = args.n_iters
    modes = args.modes
    width = args.width
    device = args.device

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
    

    # train_observations, train_actions, train_rewards, train_next_observations = data_loader.get_data()
    split = int(len(rewards) * 0.8)
    
    train_data_loader = Data(
        observations[:split],
        next_observations[:split],
        rewards[:split],
        actions[:split],
        terminals[:split],
        hist_len
    )
    
    test_data_loader = Data(
        observations[split:],
        next_observations[split:],
        rewards[split:],
        actions[split:],
        terminals[split:],
        hist_len
    )

    # train_data = (
    #     train_observations[:train_size],
    #     train_actions[:train_size],
    #     train_rewards[:train_size],
    #     train_next_observations[:train_size]
    # )

    # test_data = (
    #     train_observations[train_size:],
    #     train_actions[train_size:],
    #     train_rewards[train_size:],
    #     train_next_observations[train_size:]
    # )
    
    obs, ac, rew , n_ob = train_data_loader.get_batch(batch_size)
    print("X shape:", obs.shape)
    print("y shape:", n_ob.shape)
    print("a shape:", ac.shape)
    
    # TODO:
    # 1. make the model object
    # 2. make the optimizer object (Test with Adam lr={1e-5, 3e-5, 1e-4, 3e-4, 1e-3}})
    # 3. make the loss function object (Test with MSE and SmoothL1Loss)
    # 4. make the training loop
    # 5. make the evaluation loop
    # 6. make the plotting function (using WandB)
    
    # Additional TODOs (to be done later):
    # 1. figure out a way to compare two models (by the quality of predicted trajectories)
    # 2. need to check whether channel wise actions are enough
    # 3. need to determine whether state normalization is required
    # 4. need to determine whether batch norm is required


if __name__ == '__main__':
    args = ArgStorage({
        'learning_rate' : 1e-4,
        'weight_decay' : 1e-5,
        'n_iters' : 1000,
        'history' : 5,
        'batch_size' : 1024,
        'modes' : 9,
        'width' : 64,
        'seed' : 2023,
        'gpu_id' : 0,
    })
    
    args.device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    main(args)
