import time
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# env = gym.make('Ant-v2')
env = gym.make('FrozenLake-v1')
env.reset()

DISCOUNT = 0.9
N_ITER = 1000
TOL = 10**(-6)

convergence_track = []
valueFunction_vector = np.zeros(env.observation_space.n)

# for i in range(N_ITER):
#     random_action = env.action_space.sample()
#     returnValue = env.step(random_action)
#     # env.render()
    
#     print('Iteration : {}, Action : {}'.format(i+1, random_action))
#     print(f"State Transition Prob for {returnValue[0]} : \n {env.P[returnValue[0]][random_action]}")
#     print("-"*50)
    
#     time.sleep(2)
#     if returnValue[2]:
#         # env.reset()
#         print(returnValue)
#         break
# # print(env.P[0][1])

for iterations in range(N_ITER):
    convergence_track.append(np.linalg.norm(valueFunction_vector, 2))
    valueFunction_vector_next_iter = np.zeros(env.observation_space.n)
    
    for state in env.P:
        outerSum = 0
        
        for action in env.P[state]:
            innerSum = 0
            
            for probabiltiy, nextState, reward, isTerminal in env.P[state][action]:
                innerSum += probabiltiy * (reward + DISCOUNT * valueFunction_vector[nextState])
                
            outerSum += 0.25 * innerSum
        
        valueFunction_vector_next_iter[state] = outerSum
    
    if np.max(np.abs(valueFunction_vector_next_iter - valueFunction_vector)) < TOL:
        valueFunction_vector = valueFunction_vector_next_iter
        print('Converged')
        break
    
    valueFunction_vector = valueFunction_vector_next_iter
    
# visualize the state values
def grid_print(valueFunction,reshapeDim):
    ax = sns.heatmap(valueFunction.reshape(4,4),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.savefig('valueFunctionGrid.png',dpi=600)
    plt.show()
     
grid_print(valueFunction_vector,4)
 
plt.plot(convergence_track)
plt.xlabel('steps')
plt.ylabel('Norm of the value function vector')
plt.savefig('convergence.png',dpi=600)
plt.show()
    
env.close()

