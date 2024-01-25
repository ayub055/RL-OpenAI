import gym
import numpy as np
from envs import register_envs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback



register_envs()
# wandb.init(project='mbrl2', entity='mbrl-nfo', name='SAC_policy')
config = {
    "algorithm" : "SAC",
    "policy_type": "MlpPolicy",
    "total_timesteps": 1_000_000,
    "env_name": "cheetah-cs285-v0",
    "episode_length": 500
}



def main():
    
    name = f"{config['algorithm']}_policy_{config['episode_length']}"
    run = wandb.init(
    project="mbrl2",
    name=name,
    config=config,
    save_code=True,  # optional
    monitor_gym=True,
    sync_tensorboard=True
    )
    env = gym.make('cheetah-cs285-v0')
    env = Monitor(env)
    NUM_ENVS = 1
    N_STEPS = config['episode_length'] * NUM_ENVS
    
    model = SAC(
        policy = 'MlpPolicy',
        env = env,
        batch_size = 256,
        train_freq=1,
        gamma = 0.999,
        ent_coef = 'auto',
        verbose=1,
        device='cuda',
        tensorboard_log=f"runs/{name}")
    
    # model = PPO(
    #     policy="MlpPolicy",
    #     env=env,
    #     n_steps=N_STEPS,
    #     batch_size=256,
    #     n_epochs=4,
    #     gamma=0.999,
    #     gae_lambda=0.98,
    #     ent_coef=0.01,
    #     verbose=1,
    #     device='cuda',
    #     tensorboard_log=f"runs/{name}")
    
    model.learn(total_timesteps=1_000_000, log_interval=4, 
                callback=WandbCallback(model_save_path=f"models/{name}"))
    model.save("halfcheetah_sac")

    del model

    model = SAC.load("halfcheetah_sac")

    obs = env.reset()
    done = False
    total_reward = 0.0
    i = 0
    while not done and i < 500:
        ac, _ = model.predict(obs)  # Use the model to predict actions
        obs, rew, done, _ = env.step(ac)
        total_reward += rew
        i += 1

    print(total_reward)

    #####################
    ### Agent Evaluation
    #####################
    eval_env = Monitor(gym.make('cheetah-cs285-v0'))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    
    
    run.finish()

if __name__ == "__main__":
    main()
