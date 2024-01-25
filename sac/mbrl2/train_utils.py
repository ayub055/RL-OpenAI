import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import get_action, add_noise


def train_dynamics_model(
        model, optimizer, obs, acs, next_obs, rewards, num_steps,
        device='cuda', log_freq=10, batch_size=512, add_sl_noise=False
) -> list:
    losses = []
    for _ in range(num_steps + 1):
        batch_idxs = torch.randint(0, len(obs), (batch_size,))
        # TODO: change the batch_obs for the FNO training here
        batch_obs = torch.from_numpy(add_noise(obs[batch_idxs])).float()
        batch_n_obs =\
            torch.from_numpy(add_noise(next_obs[batch_idxs])).float()
        batch_r = torch.from_numpy(rewards[batch_idxs]).float()
        batch_acs = torch.from_numpy(acs[batch_idxs]).float()
        pred_n_s, pred_r = model(batch_obs.to(device), batch_acs.to(device))
        loss =\
            F.mse_loss(pred_n_s.squeeze(), batch_n_obs.squeeze().to(device))
        loss += F.mse_loss(pred_r.squeeze(), batch_r.squeeze().to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if log_freq is not None and _ % log_freq == 0:
            print(f"Loss after {_} steps is {loss.item()}")
        losses.append(loss.item())
    return losses


def train_policy(policy, models, env, device, num_steps, batch_size, threshold):
    # TODO:
    # 1. train the policy using the fake env (use the dynamics model)
    # 2. The termination state is that the models are in diagreement beyond a threshold
    raise NotImplementedError


def train_ensemble(
        models, optimizers, data, num_steps, device, log_freq,
        batch_size, add_sl_noise
) -> None:
    obs, acs, next_obs, rewards = data
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        losses = train_dynamics_model(
            model, optimizer, obs, acs, next_obs, rewards, num_steps,
            device, log_freq, batch_size, add_sl_noise)
        print(f'Model: {i}, Avg loss: {sum(losses) / len(losses)}')
    
    for i, model in enumerate(models):
        torch.save(model.state_dict(), f'./models/model_{i}.pt')


def evaluate_policy(env, models, device, num_steps):
    rewards = []
    for _ in tqdm(range(5)):
        done = False
        ob = env.reset()
        total_reward, steps = 0, 0
        while not done and steps < num_steps:
            ac = get_action(
                env, models, ob, device=device, num_samples=4000, future_len=15)
            next_ob, reward, done, _ = env.step(ac)
            ob = next_ob
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
    print(f"Average reward: {sum(rewards)/len(rewards)}")
    wandb.log({
        'reward0': rewards[0],
        'reward1': rewards[1],
        'reward2': rewards[2],
        'reward3': rewards[3],
        'reward4': rewards[4]
    })
