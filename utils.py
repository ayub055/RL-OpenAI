import numpy as np
import torch
import gym
import d4rl
from prettytable import PrettyTable

def compare_trajectory(model, data_loader, device, traj_len=10, is_baseline=False):
    true_trajectory = []
    predicted_trajectory = []
    
    idxs = data_loader.valid_indices
    idxs = idxs[::-1]
    idxs.append(-1)
    valid_idxs = []
    c = 0
    for idx, n_idx in zip(idxs[:-1], idxs[1:]):
        if c >= traj_len:
            valid_idxs.append(idx)
        if idx - n_idx > 1:
            c = 0
        c += 1
    
    selected_idx = np.random.choice(valid_idxs)
    obs, ac, _, n_ob = data_loader.get_batch(None, device, [selected_idx])
    input_queue = [ob for ob in obs.squeeze(0).cpu().numpy().transpose(1, 0)]

    # Use the trained model for prediction
    model.eval()

    with torch.no_grad():

        # Perform trajectory prediction for 10 time steps
        for _ in range(10):
            obs = torch.from_numpy(np.array(input_queue)).to(device).unsqueeze(0)
            obs = obs.permute(0, 2, 1)
            if is_baseline:
                obs = obs[:, :, -1]
            pred_ob = model(obs, ac).to(device)
            
            # print(f"True: {n_ob.squeeze().cpu().numpy()}, Predicted: {pred_ob.squeeze().cpu().numpy()}")
            
            true_trajectory.append(n_ob.squeeze().cpu().numpy())
            predicted_trajectory.append(pred_ob.squeeze().cpu().numpy())
            input_queue.pop(0)
            input_queue.append(pred_ob.squeeze().cpu().numpy())
            ac = torch.from_numpy(data_loader.actions[selected_idx]).to(device).unsqueeze(0)
            selected_idx += 1
            n_ob = torch.from_numpy(data_loader.next_observations[selected_idx]).to(device).unsqueeze(0)
            
            # # Convert current_state and current_action to tensors
            # current_state_tensor = torch.from_numpy(current_state) if isinstance(current_state, np.ndarray) else current_state
            # current_state_tensor = current_state_tensor.to(device)

            # current_action_tensor = torch.from_numpy(current_action) if isinstance(current_action, np.ndarray) else current_action
            # current_action_tensor = current_action_tensor.to(device)

            # # Predict the next state
            # next_state_hat = model(current_state_tensor, current_action_tensor)
            # # print(f"Next state: {next_state_hat.shape}")

            # # Append the true and predicted states to the respective lists
            # true_trajectory.append(current_state.cpu().numpy())
            # predicted_trajectory.append(next_state_hat.cpu().numpy())

            # current_state_tensor = torch.cat((current_state_tensor[:, :, 1:], next_state_hat), dim=-1).cpu().numpy()
            # print(f"Current : {current_state_tensor.shape}")

    # Calculate MAPE for each time step
    # true_trajectory = np.array(true_trajectory)
    # predicted_trajectory = np.array(predicted_trajectory)
    
    mape_values = np.mean(np.abs((np.array(true_trajectory) - np.array(predicted_trajectory)) / np.array(true_trajectory)) * 100)

    return mape_values

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == "__main__":
    from dataloader_delta import Data, Baseline2
    from dataloader import FNO1d
    
    env = gym.make('halfcheetah-medium-v2')
    data = env.get_dataset()
    observations = data['observations']
    actions = data['actions']
    next_observations = data['next_observations']
    rewards = data['rewards']
    timeouts = data['timeouts']
    terminals = np.logical_or(timeouts, data['terminals'])
    hist_len = 5
    device = torch.device("cpu")
    
    split = int(0.8 * len(observations))
    test_data_loader = Data(
        observations[split:],
        next_observations[split:],
        rewards[split:],
        actions[split:],
        terminals[split:],
        hist_len,
        device
    )
    
    model_baseline = Baseline2(17, 6, 512)
    model_baseline.load_state_dict(torch.load("model_Delta-halfcheetah_lr_0.003_hidden_512.pth"))
    
    print("Baseline:")
    count_parameters(model_baseline)
    
    model_fno = FNO1d(9, 256, 5, 6, device)
    model_fno.load_state_dict(torch.load("model_FNO-halfcheetah_lr0.001_width_256_NO_BN.pth"))
    
    print("FNO:")
    count_parameters(model_fno)
    
    mape_baseline = compare_trajectory(model_baseline, test_data_loader, device, is_baseline=True)
    mape_fno = compare_trajectory(model_fno, test_data_loader, device)
    
    print(f"MAPE Baseline: {mape_baseline}")
    print(f"MAPE FNO: {mape_fno}")
