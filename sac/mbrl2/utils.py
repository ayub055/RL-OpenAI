import gym
import copy
import torch
import numpy as np
from typing import Optional, Tuple, List
from tqdm import tqdm
from typing_extensions import TypedDict

class ArgParser:
    def __init__(self, args: dict) -> None:
        self.__dict__.update(args)
        
class PathDict(TypedDict):
    observation: np.ndarray
    image_obs: np.ndarray
    reward: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray

def load_models(models: List[torch.nn.Module], 
                model_path: str) -> None:
    for i, model in enumerate(models):
        model.load_state_dict(torch.load(f"{model_path}/model_{i}.pt"))
        model.train()
        
        
def concat_data(
    data1: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    data2: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs1, acs1, next_obs1, rewards1 = data1
    obs2, acs2, next_obs2, rewards2 = data2
    obs = np.concatenate([obs1, obs2])[-200_000:]
    acs = np.concatenate([acs1, acs2])[-200_000:]
    next_obs = np.concatenate([next_obs1, next_obs2])[-200_000:]
    rewards = np.concatenate([rewards1, rewards2])[-200_000:]
    return obs, acs, next_obs, rewards


def get_expert_data(
        env: gym.Env, 
        num_samples: int = None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    data = env.get_dataset()
    if num_samples is not None:
        indices = np.random.choice(len(data['observations']),
                                   size=num_samples, replace=False)
        obs = np.array(data['observations'][indices])
        acs = np.array(data['actions'][indices])
        rewards = np.array(data['rewards'][indices])
        next_obs = np.array(data['next_observations'][indices])
    else:
        obs = np.array(data['observations'])
        acs = np.array(data['actions'])
        rewards = np.array(data['rewards'])
        next_obs = np.array(data['next_observations'])
    
    print(obs.shape, acs.shape, next_obs.shape, rewards.shape)
    return obs, acs, next_obs, rewards
    
def get_action(
        env: gym.Env,
        models: List[torch.nn.Module],
        ob: np.ndarray,
        num_samples: int = 1000,
        future_len: int = 10,
        device: torch.device = torch.device('cuda'),
) -> np.ndarray:
    obs = torch.from_numpy(ob).float().reshape(1, -1).repeat(num_samples, 1)
    obs = obs.to(device)
    acs = []
    rewards = torch.zeros(num_samples, 1).to(device)
    gamma = 1.0
    for _ in range(num_samples):
        ac = env.action_space.sample()
        acs.append(ac)
    acs = torch.from_numpy(np.array(acs)).float().to(device)
    acs_bak = copy.deepcopy(acs)
    for _ in range(future_len):
        n_obs, rs = [], []
        for model in models:
            with torch.no_grad():
                n_ob, r = model(obs, acs)
            n_obs.append(n_ob)
            rs.append(r)
        n_obs = torch.cat(n_obs, dim=-1).reshape(num_samples, -1, len(models)).mean(dim=-1)
        rs = torch.cat(rs, dim=-1).reshape(num_samples, -1, len(models)).mean(dim=-1)
        rewards += rs * gamma
        gamma *= 0.99
        obs = n_obs
        permuted_idxs = torch.randperm(num_samples)
        acs = acs[permuted_idxs]
    ac = acs_bak[torch.argmax(rewards)]
    return ac.cpu().numpy()


def collect_data(
        env: gym.Env,
        models: Optional[List[torch.nn.Module]],
        data_size: int = 5000,
        collect_randomly: bool = False,
        epsilon: float = 0.1,
        future_len: int = 15,
        device: torch.device = torch.device('cuda'),
        episode_len: int = 150,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs, acs, next_obs, rewards = [], [], [], []
    if collect_randomly:
        while len(obs) < data_size:
            done, steps = False, 0
            ob = env.reset()
            while not done and steps < episode_len:
                ac = env.action_space.sample()
                next_ob, reward, done, _ = env.step(ac)
                obs.append(ob)
                acs.append(ac)
                next_obs.append(next_ob)
                rewards.append(reward)
                ob = next_ob
                steps += 1
        obs = np.array(obs)
        acs = np.array(acs)
        next_obs = np.array(next_obs)
        rewards = np.array(rewards)
    else:
        while len(obs) < data_size:
            done, steps = False, 0
            ob = env.reset()
            while not done and steps < episode_len:
                # TODO: change this to get action from learned policy and we use gaussian noise for exploration
                if models is None or np.random.rand() < epsilon:
                    ac = env.action_space.sample()
                else:
                    ac = get_action(
                        env, models, ob, future_len=future_len,
                        num_samples=4000, device=device
                    )
                next_ob, reward, done, _ = env.step(ac)
                obs.append(ob)
                acs.append(ac)
                next_obs.append(next_ob)
                rewards.append(reward)
                ob = next_ob
                if len(obs) % 200 == 0:
                    print(f'Collected {len(obs)} samples')
                steps += 1
    return obs, acs, next_obs, rewards


def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp)  # (num data points, dim)

    mean_data = np.mean(data, axis=0)  # mean of data

    # if mean is 0, make it 1e-8 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 1e-8

    # width of normal distribution to sample noise from
    # larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data

def get_offline_data(env):
    '''
    For mbrl to load dataset and return 
    1. trajectories
    2. total time steps take (not relevant in the CS285 code)
    '''
    data = env.get_dataset()
    
    obs = np.array(data['observations'])
    acs = np.array(data['actions'])
    rewards = np.array(data['rewards'])
    next_obs = np.array(data['next_observations'])
    terminals = np.array(data['terminals'])
    timeouts = np.array(data['timeouts'])
    terminals = np.logical_or(terminals, timeouts)

    print(obs.shape, acs.shape, next_obs.shape, rewards.shape, terminals.shape)
    
    paths: List[PathDict] = []
    path: PathDict = {
        'observation': [],
        'reward': [],
        'action': [],
        'next_observation': []}
    
    envsteps_this_batch = 0
    for i in range(len(obs)):
        if not terminals[i]:
            path['observation'].append(obs[i])
            path['reward'].append(rewards[i])
            path['action'].append(acs[i])
            path['next_observation'].append(next_obs[i])    
        else:
            if len(path['observation']) > 0:
                paths.append(path)
                envsteps_this_batch += len(path['observation'])
            
            path = {
                'observation': [],
                'reward': [],
                'action': [],
                'next_observation': []
                }
    if len(path['observation']) > 0:
        paths.append(path)
        envsteps_this_batch += len(path['observation'])
        
    print(f"Total trajectories: {len(paths)}, Total envsteps: {envsteps_this_batch}")
    print(np.array(paths[1]['observation']).shape)
    return paths, envsteps_this_batch
