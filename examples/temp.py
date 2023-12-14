import numpy as np
import torch
import gym
import d4rl


# def main():
#     env = gym.make('halfcheetah-medium-v2')
#     dataset = env.get_dataset()
#     observations = dataset['observations']
#     actions = dataset['actions']
#     rewards = dataset['rewards']
#     terminals = dataset['terminals']
#     timeouts = dataset['timeouts']
#     terminals = np.logical_or(terminals, timeouts)
#     next_observations = dataset['next_observations']
#     print(observations.shape, actions.shape, rewards.shape, terminals.shape, next_observations.shape)
#     print(observations[0], actions[0], rewards[0], terminals[0], next_observations[0])
#     print(observations[1], actions[1], rewards[1], terminals[1], next_observations[1])
#     print(observations[2], actions[2], rewards[2], terminals[2], next_observations[2])

#     print(observations[999], actions[999], rewards[999], terminals[999], next_observations[999])
#     print(observations[1000], actions[1000], rewards[1000], terminals[1000], next_observations[1000])
#     print(observations[1001], actions[1001], rewards[1001], terminals[1001], next_observations[1001])


# if __name__ == '__main__':
#     main()
def get_data(env, history):
    dataset = env.get_dataset()
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    timeouts = dataset['timeouts']
    terminals = np.logical_or(terminals, timeouts)
    next_observations = dataset['next_observations']

    # Assuming observations and next_observations are NumPy arrays from your dataset
    # Make sure to reshape the data if needed
    observation_dim = observations.shape[1]  # Update this based on your observation dimensionality
    observations = observations.reshape(-1, observation_dim)
    next_observations = next_observations.reshape(-1, observation_dim)

    # Convert to PyTorch tensors
    observations_tensor = torch.Tensor(observations)
    next_observations_tensor = torch.Tensor(next_observations)

    split = 0.8
    split_index = int(len(observations_tensor) * split)

    train_obs = observations_tensor[:split_index]
    test_obs = observations_tensor[split_index:]

    train_next_obs = next_observations_tensor[:split_index]
    test_next_obs = next_observations_tensor[split_index:]

    # Convert to the desired format (batch_size, x=64, c=11)
    train_data = train_obs.reshape(-1, 64, 11)
    test_data = test_obs.reshape(-1, 64, 11)

    return train_data, test_data

def main():
    env = gym.make('halfcheetah-medium-v2')
    # dataset = env.get_dataset()
    history = 10
    train_data, test_data = get_data(env, history)
    print(train_data.shape, test_data.shape)
    
    
    # print(observations.shape, actions.shape, rewards.shape, terminals.shape, next_observations.shape)
    # print(observations[0], actions[0], rewards[0], terminals[0], next_observations[0])
    # print(observations[1], actions[1], rewards[1], terminals[1], next_observations[1])
    # print(observations[2], actions[2], rewards[2], terminals[2], next_observations[2])

    # print(observations[999], actions[999], rewards[999], terminals[999], next_observations[999])
    # print(observations[1000], actions[1000], rewards[1000], terminals[1000], next_observations[1000])
    # print(observations[1001], actions[1001], rewards[1001], terminals[1001], next_observations[1001])

    # Assuming observations and next_observations are NumPy arrays from your dataset
    # Make sure to reshape the data if needed
    # observation_dim = observations.shape[1]  # Update this based on your observation dimensionality
    
    # print(observation_dim)
    # observations = observations.reshape(-1, observation_dim)
    # print(observations)
    # next_observations = next_observations.reshape(-1, observation_dim)

    # # Convert to PyTorch tensors
    # observations_tensor = torch.Tensor(observations)
    # next_observations_tensor = torch.Tensor(next_observations)


    # # Define the FNO1d model
    # modes = 8  # Adjust based on your requirements
    # width = 64  # Adjust based on your requirements
    # history = 10  # Adjust based on your requirements
    # model = FNO1d(modes, width, history)

    # # Initialize the model, loss function, and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Training the model
    # num_epochs = 50
    # batch_size = 64

    # train_dataset = TensorDataset(train_obs, train_next_obs)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # for epoch in range(num_epochs):
    #     for batch_obs, batch_next_obs in train_loader:
    #         optimizer.zero_grad()
    #         predictions = model(batch_obs)
    #         loss = criterion(predictions, batch_next_obs)
    #         loss.backward()
    #         optimizer.step()

    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # # Evaluate the model on the test set
    # test_predictions = model(test_obs)
    # test_loss = criterion(test_predictions, test_next_obs)
    # print(f"Test Loss: {test_loss.item()}")

if __name__ == '__main__':
    main()