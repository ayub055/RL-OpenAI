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
from utils import compare_trajectory, count_parameters


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
    # def __init__(self, conv_channels, state_dim, ac_dim):
    def __init__(self, conv_channels, ac_dim):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(conv_channels, conv_channels, 1)
        # self.bn = torch.nn.BatchNorm1d(conv_channels)  # TODO: check is this is even required
        self.ac_dim = ac_dim
        # self.ac_encoder = nn.Linear(ac_dim, conv_channels * state_dim)  # TODO: make it general (state dim independent)
        self.ac_encoder = nn.Linear(ac_dim, conv_channels)
    
    def forward(self, x, a):
        _, channels_, _ = x.shape
        x = self.conv(x) # 32 x 20 x 17
        a = self.ac_encoder(a).reshape(-1, channels_, 1) # 32 x 20 x 1
        x = x + a
        # x = self.bn(x) # 32 x 20 x 17
        return x


class FNO1d(nn.Module):
    # def __init__(self, modes, width, history, state_dim, ac_dim, device):
    def __init__(self, modes, width, history, ac_dim, device):
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
        self.device = device
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d_fast(self.width, self.width, self.modes)
        # self.w0 = ConvBlock(width, state_dim, ac_dim)
        # self.w1 = ConvBlock(width, state_dim, ac_dim)
        # self.w2 = ConvBlock(width, state_dim, ac_dim)
        # self.w3 = ConvBlock(width, state_dim, ac_dim)
        self.w0 = ConvBlock(width, ac_dim)
        self.w1 = ConvBlock(width, ac_dim)
        self.w2 = ConvBlock(width, ac_dim)
        self.w3 = ConvBlock(width, ac_dim)

        self.fc1 = nn.Linear(self.width, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, a): # x is state, a is action
        grid = self.get_grid(x.shape, x.device)
        
        # x = x.to(self.device)
        # a = a.to(self.device)
        x = torch.cat((x, grid), dim=-1)  # 32 x 17 x (10+1)
        # print(x.shape)
        x = self.fc0(x)  # 32 x 17 x 20
        # print(x.shape)
        
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
        # self.bn1 = nn.BatchNorm1d(128)
        x = self.fc2(x)
        return x.to(self.device)

    def get_grid(self, shape, device):
        batch_size, size_x = shape[0], shape[1]
        grid_x = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        grid_x = grid_x.reshape(1, size_x, 1).repeat([batch_size, 1, 1])
        return grid_x

def main(args):
    batch_size = args.batch_size
    hist_len = args.history
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    n_iters = args.n_iters
    modes = args.modes
    width = args.width
    device = args.device
    print(f"using : {device}")
    
    # model = FNO1d(
    #     modes=modes,
    #     width=width,
    #     history=hist_len,
    #     ac_dim=6,
    #     device=device
    # )
    # count_parameters(model)
    # import sys; sys.exit()

    # Generate experiment name based on learning rate and loss function
    
    experiment_name = f"FNO-halfcheetah_lr_{args.learning_rate}_width_{args.width}_NOBN_NO_STATE_modes_{args.modes}"
    wandb.init(project="mbrl-nfo", group ="State-INDEPENDENT-No-BN", name=experiment_name)
    wandb.config.update(args)
    
    env = gym.make('halfcheetah-medium-v2')
    dataset = env.get_dataset()
    
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
        hist_len,
        device
    )
    
    test_data_loader = Data(
        observations[split:],
        next_observations[split:],
        rewards[split:],
        actions[split:],
        terminals[split:],
        hist_len,
        device
    )

    
    obs, ac, rew , n_ob = train_data_loader.get_batch(batch_size, device)
    print("X shape:", obs.shape)
    print("y shape:", n_ob.shape)
    print("a shape:", ac.shape)
    
    
    # print(obs[0,:,:])
    # print(n_ob[0,:,:])
    
    
    # print(obs[1,:,:])
    # print(n_ob[1,:,:])
    # print(n_ob)
    # print(obs[1,:,:])
    # print(n_ob[0,:,:])
    
    # print(n_ob[0,:,:])
    
    
    # import sys; sys.exit()
    # TODO:
    # 1. make the model object
    model = FNO1d(
        modes=modes,
        width=width,
        history=hist_len,
        # state_dim=observations.shape[1],
        ac_dim=actions.shape[1],
        device=device
    )
    
    # 2. make the optimizer object (Test with Adam lr={1e-5, 3e-5, 1e-4, 3e-4, 1e-3}})
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    if args.loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss == 'SmoothL1':
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError("Invalid loss function. Use 'MSE' or 'SmoothL1'.")

    # 4. make the training loop
    model.to(device)
    count_parameters(model)
    
    # import sys;sys.exit()
    model.train()
    
    hist = np.inf
    for i in range(n_iters):
        X, ac, _, y = train_data_loader.get_batch(batch_size, device)
        X, y = X.to(device), y.to(device)
        y_hat = model(X, ac)
        
        loss = loss_fn(y_hat, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            with torch.no_grad():
                test_X, test_ac, _, test_y = test_data_loader.get_batch(batch_size, device)
                test_X, test_y = test_X.to(device), test_y.to(device)
                test_y_hat = model(test_X, test_ac)
                val_loss = loss_fn(test_y_hat, test_y)
                print(i, ", Train Error:", loss.item(), ", Test Error:", val_loss.item())
                # wandb.log({"Training Loss": loss.item()})
                wandb.log({"Train error": loss.item(), "Test error":val_loss.item()})
                
                if val_loss.item() < hist:
                    model_filename = f'model_{experiment_name}.pth'
                    torch.save(model.state_dict(), model_filename)
                    # torch.save(model.state_dict(), 'model_check.pth')
                    hist = val_loss.item()
                    
        model.train()
    # model.load_state_dict(torch.load('model_check.pth'))
    # mape = compare_trajectory(model, test_data_loader, device, batch_size)
    # print("MAPE values for each time step:", mape.shape)
       
    # Additional TODOs (to be done later):
    # 0. use a simple baseline (2 layer MLP with 512 hidden units) and compare the performance
    # 1. figure out a way to compare two models (by the quality of predicted trajectories)
    #  - Given a state, and an action, predict the next state. Now, use this predicted state as an input to the model again, and predict the next state given the action that was taken by the policy. Do it for 10 length trajectory and compare the two trajectories.
    # 2. need to check whether channel wise actions are enough
    # 3. need to determine whether state normalization is required
    # 4. need to determine whether batch norm is required
    # 5. Do we need resuidue model?

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_iters', type=int, default=20000)
    parser.add_argument('--history', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--modes', type=int, default=5)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--loss', choices=['MSE', 'SmoothL1'], default='SmoothL1')

    args = parser.parse_args()

    
    args.device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    main(args)


