import argparse
import gym
import numpy as np
from envs import register_envs
from stable_baselines3 import PPO, SAC
from tqdm import tqdm
from typing_extensions import TypedDict, List, Tuple

register_envs()

class PathDict(TypedDict):
    observation: np.ndarray
    image_obs: np.ndarray
    reward: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    terminal: np.ndarray
    
def Path(
    obs: List[np.ndarray],
    image_obs: List[np.ndarray],
    acs: List[np.ndarray],
    rewards: List[np.ndarray],
    next_obs: List[np.ndarray], 
    terminals: List[bool],
) -> PathDict:
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def load_data(file_path='./transitions_data.npz'):
    data = np.load(file_path)
    observations = data['observations']
    next_observations = data['next_observations']
    rewards = data['rewards']
    actions = data['actions']
    terminals = data['terminals']

    
    print("in load data")
    print(observations.shape, actions.shape, next_observations.shape, rewards.shape, terminals.shape)
    
    return observations, actions, next_observations, rewards, terminals

def save_transitions(transitions, file_path):
    np.save(file_path, transitions)
    
def sample_offline_trajectory(
    env,
    model,
    episode_len: int = 500
) -> PathDict:

    ob = env.reset()
    
    done = False
    steps = 0
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    num_ep_transitions = 0
    while True:
        ac, _ = model.predict(ob)
        # print(ac.shape)
        n_ob, rew, done, _ = env.step(ac)
        steps += 1
        acs.append(ac)

        obs.append(ob)
        next_obs.append(n_ob)
        rewards.append(rew)
        ob = n_ob
        num_ep_transitions += 1

        if done or steps == episode_len:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_offline_trajectories(
    env,
    model_path,
    episode_len: int = 500,
    max_transitions: int = 5_00_000
) -> Tuple[List[PathDict], int]:
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.
    """
    model = SAC.load(model_path)
    env = gym.make(env)
    
    num_transitions = 0
    paths: List[PathDict] = []
    with tqdm (total = max_transitions) as pbar:
        while num_transitions < max_transitions:
            path: PathDict = sample_offline_trajectory(env, model, episode_len)
            paths.append(path)
            num_transitions += path['observation'].shape[0]
            pbar.update(path['observation'].shape[0])

    return paths, num_transitions

def simulate_environment(model_path, env_name, episode_length, total_transitions):

    model = SAC.load(model_path)
    env = gym.make(env_name)

    collected_transitions = 0
    trajectories = []
    steps = 0

    with tqdm(total = total_transitions) as pbar:
        
        observations, next_observations, rewards, actions, terminals = [], [], [], [], []
        while collected_transitions < total_transitions:
            obs = env.reset()
            done = False
            steps = 0
            while True:
                action, _ = model.predict(obs)
                next_obs, reward, done, _ = env.step(action)
                steps += 1
                
                observations.append(obs)
                next_observations.append(next_obs)
                rewards.append(reward)
                actions.append(action)
                obs = next_obs
                collected_transitions += 1
                if done or steps == episode_length:
                    terminals.append(1)
                    break
                else:
                    terminals.append(0)
                pbar.update(1)
                
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminals = np.array(terminals)
            
    save_path = './transitions_data.npz'
    np.savez(save_path,
             observations=observations,
             next_observations=next_observations,
             rewards=rewards,
             actions=actions,
             terminals= terminals)
            
    print(observations.shape, actions.shape, next_observations.shape, rewards.shape, terminals.shape)
    return observations, actions, next_observations, rewards, terminals


def get_offline_data(min_transitions=5_00_000):
            '''
            For mbrl to load dataset and return 
            1. trajectories
            2. total time steps take (not relevant in the CS285 code)
            '''
            train_video_paths = None # TODO: Later on add functionatlity
            # data = load_data
            
            obs, acs, next_obs, rewards, terminals = load_data('./transitions_data.npz')
            print("Hey")
            print(obs.shape, acs.shape, next_obs.shape, rewards.shape, terminals.shape)
            print("Hey")
            # import sys; sys.exit()
            paths: List[PathDict] = []
            path: PathDict = {
                'observation': [],
                'reward': [],
                'action': [],
                'next_observation': [],
                'terminal': []}
            
            envsteps_this_batch = 0
            total_transitions = 0
            
            print(len(obs))
            for i in range(len(obs)):
                
                # print(f"Iteration : {i}, {terminals[i]}")
                if not terminals[i]:
                    path['observation'].append(obs[i])
                    path['reward'].append(rewards[i])
                    path['action'].append(acs[i])
                    path['next_observation'].append(next_obs[i])
                    path['terminal'].append(terminals[i]) 
                     
                    total_transitions += 1
                    if total_transitions == min_transitions:
                        paths.append(path)
                        envsteps_this_batch += len(path['observation'])
                        break
                else:
                    if len(path['observation']) > 0:
                        paths.append(path)
                        envsteps_this_batch += len(path['observation'])
                    
                    path = {
                        'observation': [],
                        'reward': [],
                        'action': [],
                        'next_observation': [],
                        'terminal': []
                        }
                
            print(f"Total trajectories: {len(paths)}, Total envsteps: {envsteps_this_batch}")
            print(np.array(paths[1]['observation']).shape)
            return paths, envsteps_this_batch, train_video_paths
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--env_name', type=str, default='cheetah-cs285-v0')
    parser.add_argument('--episode_length', type=int, default=500)
    parser.add_argument('--total_transitions', type=int, default=500000)

    args = parser.parse_args()

    # observations, actions, next_observations, rewards, terminals = simulate_environment(args.model_path, args.env_name, args.episode_length, args.total_transitions)
    
    paths, transitions = sample_trajectories(args.env_name, args.model_path, max_transitions=5_00_000)
    for path in paths:
        print("obs: ", path['observation'].shape)
        print("n_obs: ", path['next_observation'].shape)
        print("acs: ", path['action'].shape)
        print("rew: ", path['reward'].shape)
        print("terminal: ", path['terminal'].shape)
        
        break
    
    # get_offline_data()

    # TODO: Process the collected trajectories as needed 
    # print(f"{args.total_transitions} transitions collected successfully.")
    # print(len(observations))

if __name__ == "__main__":
    main()
