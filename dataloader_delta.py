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

############################################################            
#################### BaseLine ##############################
############################################################

class Baseline1(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Baseline1, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)  # Concatenate state and action along the last dimension
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x)        
        x = x.unsqueeze(-1) #([1024, 17, 1])
        return x
    

# class Baseline2(nn.Module):
#     def __init__(self, delta_predictor):
#         super(Baseline2, self).__init__()
#         self.delta_predictor = delta_predictor

#     def forward(self, state, action):
#         # Get the predicted delta
#         state, delta_pred = self.delta_predictor(state, action)

#         # Compute the next predicted state
#         next_state = state + delta_pred

#         return next_state, delta_pred
    
    
    
# class DeltaPredictor(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim):
#         super(DeltaPredictor, self).__init__()
#         self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, state_dim)
    
#     def forward(self, state, action):
#         # state_t, state_t_1 = state[:, :, -1], state[:, :, -2]
#         # change_in_state = state_t - state_t_1   #ground truth
#         state_t = state[:, :, -1]
#         input = torch.cat((state_t, action), dim=1)
#         delta_pred = self.fc2(torch.relu(self.fc1(input)))
#         # print(delta_pred.shape)
#         # print(state_t.shape)
#         # next_state = state_t + delta_pred
#         # next_state = next_state.unsqueeze(-1)
#         # print(next_state.shape)
#         return state_t, delta_pred # should return this too, delta_pred


def main(args):
    batch_size = args.batch_size
    hist_len = args.history
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    n_iters = args.n_iters
    modes = args.modes
    width = args.width
    device = args.device
    model = args.model
    print(f"using : {device}")

    # Generate experiment name based on learning rate and loss function
    # experiment_name = f"FNO-halfcheetah-medium-v2_lr{args.learning_rate}_loss{args.loss}"
    # experiment_name = f"Baseline-halfcheetah-medium-v2_model_{args.model}_{args.loss}"

    # wandb.init(project="mbrl-nfo", name=experiment_name)
    # wandb.config.update(args)
    
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
    print(observations.shape)
    # print(actions.shape[1])
    
    # obs_mu, obs_std = np.mean(observations), np.std(observations)
    # observations = (observations - obs_mu) / obs_std
    # next_observations = (next_observations - obs_mu) / obs_std

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
    
    
    if args.model == 'baseline':
        model = Baseline1(
            state_dim=observations.shape[1],
            action_dim=actions.shape[1],
            hidden_dim=512
        ).to(device)
        
    elif args.model == 'FNO1d':
            model = FNO1d(
            modes=modes,
            width=width,
            history=hist_len,
            state_dim=observations.shape[1],
            ac_dim=actions.shape[1],
            device=device
        )
    elif args.model == 'dynamic':
        model = Baseline1(
            state_dim=observations.shape[1],
            action_dim=actions.shape[1],
            hidden_dim=512
        ).to(device)
        # state, n_state = obs[:, :, -1], y
        # change_in_state = n_state - state   #ground truth
        
        # pred_change_in_state = model(state, ac)
        # loss = F.smooth_l1_loss(delta_pred, change_in_state)
        # print(loss)
    else:
        raise ValueError("Invalid model. Use 'baseline' or 'FNO1d'.")
 
    # import sys; sys.exit()
    # import sys;sys.exit()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    # 3. make the loss function object (Test with MSE and SmoothL1Loss)
    if args.loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss == 'SmoothL1':
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError("Invalid loss function. Use 'MSE' or 'SmoothL1'.")

    
    # 4. make the training loop
    model.to(device)
    model.train()
    
    hist = np.inf
    for i in range(n_iters + 1):
        X, ac, _, y = train_data_loader.get_batch(batch_size, device)
        X, y = X.to(device), y.to(device)
        # y_hat = model(X, ac)
        state, n_state = X[:, :, -1], y.squeeze()
        change_in_state = n_state - state   #ground truth
        
        pred_change_in_state = model(state, ac).squeeze()
        loss = F.smooth_l1_loss(pred_change_in_state, change_in_state)

        # loss = loss_fn(y_hat, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            with torch.no_grad():
                test_X, test_ac, _, test_y = test_data_loader.get_batch(batch_size, device)
                test_X, test_y = test_X.to(device), test_y.to(device)
                # test_y_hat = model(test_X, test_ac)
                
                #####Specific for dynamic model #######
                ########################################
                state, n_state = test_X[:, :, -1], test_y.squeeze()
                change_in_state = n_state - state   #ground truth
                
                pred_change_in_state = model(state, test_ac).squeeze()
                val_loss = F.smooth_l1_loss(pred_change_in_state, change_in_state)
                ########################################
                ########################################
                # val_loss = loss_fn(test_y_hat, test_y)
                
                print(i, ", Train Error:", loss.item(), ", Test Error:", val_loss.item())
                # wandb.log({"Training Loss": loss.item()})
                
                if val_loss.item() < hist:
                    # model_filename = f'model_{experiment_name}.pth'
                    # torch.save(model.state_dict(), model_filename)
                    torch.save(model.state_dict(), 'model.pth')
                    hist = val_loss.item()
        model.train()
        
 
    # 6. make the plotting function (using WandB)
    
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
    # args = ArgStorage({
    #     'learning_rate' : 1e-4,
    #     'weight_decay' : 1e-5,
    #     'n_iters' : 1000,
    #     'history' : 5,
    #     'batch_size' : 1024,
    #     'modes' : 9,
    #     'width' : 64,
    #     'seed' : 2023,
    #     'gpu_id' : 0,
    # })
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4) # TODO: what is the best learning rate
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--n_iters', type=int, default=1000) # TODO: where to stop
    parser.add_argument('--history', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--modes', type=int, default=9)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--loss', choices=['MSE', 'SmoothL1'], default='SmoothL1')
    parser.add_argument('--model', choices=['baseline', 'FNO1d', 'dynamic'])

    args = parser.parse_args()

    
    args.device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    main(args)


