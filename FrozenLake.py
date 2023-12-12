import gymnasium as gym
import time

# env = gym.make('Ant-v2')
env = gym.make('FrozenLake-v1')
env.reset()

n_iter = 30
for i in range(n_iter):
    random_action = env.action_space.sample()
    returnValue = env.step(random_action)
    
    
    # env.render()
    
    print('Iteration : {}, Action : {}'.format(i+1, random_action))
    print(f"State Transition Prob for {returnValue[0]} : \n {env.P[returnValue[0]][random_action]}")
    print("-"*50)
    
    time.sleep(2)
    if returnValue[2]:
        # env.reset()
        print(returnValue)
        break
# print(env.P[0][1])
env.close()

# rew = 0
# while not done:
#     ac = env.action_space.sample()
#     _, r, done, _, _ = env.step(ac)
#     rew += r
#     print(rew)
# print(rew)
