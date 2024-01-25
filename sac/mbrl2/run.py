import gym
import d4rl
import torch
import wandb
import random
import numpy as np

from utils import *
from models import *
from train_utils import *


def train(
        models: list,
        optimizers: list,
        env: gym.Env,
        args: ArgParser,
        expert_train: bool,
        data  
) -> None:
    # Run evaluation before training
    # print('Evaluating random policy...')
    # evaluate_policy(env, models, args.device, args.episode_len)
    
    # collect data using random policy (initially)
    # if not expert_train:
    #     print("Collecting data through expert pretrained model...")
    #     data = collect_data(
    #         env, models, args.data_size * 4, False, args.epsilon,
    #         args.future_len, args.device, args.episode_len
    #     )
    
    for i in range(args.num_epochs):
        print('-' * 50)
        print(f'Epoch: {i}')
        print('-' * 50)
        print('Training dynamics model...')
        
        # train policy using collected data and trained dynamics model
        # NOTE: Using random shooting (for now)
        # train_policy(policy, models, env, device, num_steps, batch_size, threshold)
        
        # collect data using trained policy in eps-greedy fashion
        # print('-' * 50)
        # print('Collecting data using epsilon greedy trained policy...')
        # new_data = collect_data(
        #     env, models, args.data_size, False, args.epsilon,
        #     args.future_len, args.device, args.episode_len
        # )
        # data = concat_data(data, new_data)
        
        ########################################
        # Using data collected from SAC policy
        ########################################
        data = np.load('./transitions_data.npy', allow_pickle=True)
        print(f'Collected {len(data)} samples')
        print(f'Collected {len(data[0])} samples')
        
        
        # train dynamics model using collected data
        train_ensemble(
            models, optimizers, data, args.num_steps, args.device,
            args.log_freq, args.batch_size, args.add_sl_noise)
        
        # evaluate policy
        print('-' * 50)
        print('Evaluating dynamics model...')
        evaluate_policy(env, models, args.device, args.episode_len)


def main(args: ArgParser):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    env = gym.make(args.env_name)
    num_samples = None if args.expert_train else 50_000
    data = get_expert_data(env, num_samples)
    
    models, optimizers = get_dynamics_ensemble(
        args.ensemble_size, env, args.hidden_size, args.device,
        args.learning_rate, args.weight_decay
    )
    
    if not args.expert_train:
        t = copy.deepcopy(models[0].state_dict())
        load_models(models, args.model_path)
        print("shit!" if torch.allclose(t['fc1.weight'], models[0].state_dict()['fc1.weight']) == True else "models loaded properly")
    train(models, optimizers, env, args, args.expert_train, data)
    # train(models, optimziers, env, args)


if __name__ == "__main__":
    args = ArgParser({
        'env_name': 'halfcheetah-medium-v2',
        'model': 'MLP',
        'hidden_size': 512,
        'activation': 'relu',
        'ensemble_size': 4,
        'num_epochs': 200,
        'num_steps': 500,
        'log_freq': None,
        'data_size': 5000,
        'batch_size': 512,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'episode_len': 500,
        'future_len': 10,
        'threshold': 0.001,
        'epsilon': 0.001,
        'add_sl_noise': True,
        'expert_train':False,
        'model_path': './_pretrained_models',
        'seed': 2024,
        'device': 'cuda'
    })
    
    wandb.init(
        project='mbrl2',
        entity='mbrl-nfo',
        name='random_dynamics_halfcheetah',
        config=args.__dict__
    )
    
    main(args)
