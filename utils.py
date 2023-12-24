import numpy as np
import torch
import gym
import d4rl
from prettytable import PrettyTable



def get_selected_idx(data_loader, traj_len):
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

    return np.random.choice(valid_idxs)


def compare_trajectory(model, data_loader, device, traj_len=10, is_baseline=False, selected_idx=None):
    true_trajectory = []
    predicted_trajectory = []
    
    if selected_idx is None:
        selected_idx = get_selected_idx(data_loader, traj_len)
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
    
    # print(np.array(true_trajectory).shape)
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
    model_baseline.load_state_dict(torch.load("./saved_models/model_Delta-halfcheetah_lr_0.003_hidden_512.pth"))
    
    print("Baseline:")
    count_parameters(model_baseline)
    
    model__AR = FNO1d(5, 256, 5, 6, device)
    model__AR.load_state_dict(torch.load("model_FNO-halfcheetah_lr_0.001_width_256_modes_5_AR.pth"))
    
    
    model_fno = FNO1d(5, 256, 5, 6, device)
    model_fno.load_state_dict(torch.load("./saved_models/model_FNO-halfcheetah_lr_0.001_width_256_NOBN_NO_STATE_modes_5.pth"))
    
    print("FNO AR:")
    count_parameters(model__AR)
    
    print("FNO:")
    count_parameters(model_fno)    
    
    baseline_mapes, fno_mapes, AR_mapes  = [], [], []
    from tqdm import tqdm
    for _ in tqdm(range(1000)):
        idx = get_selected_idx(test_data_loader, 10)
        baseline_mapes.append(
            compare_trajectory(model_baseline, test_data_loader, device, is_baseline=True, selected_idx=idx))
        fno_mapes.append(
            compare_trajectory(model_fno, test_data_loader, device, selected_idx=idx))
        AR_mapes.append(
            compare_trajectory(model__AR, test_data_loader, device, selected_idx=idx))
    
    print(f"MAPE Baseline: {sum(baseline_mapes)/len(baseline_mapes)}")
    print(f"MAPE FNO: {sum(fno_mapes)/len(fno_mapes)}")
    print(f"MAPE AR: {sum(AR_mapes)/len(AR_mapes)}")
    
class EnsembleModel:
    def __init__(self, model_list, test_data_loader, batch_size, device):
        self.device = device
        self.models = model_list
        self.test_data_loader = test_data_loader
        self.batch_size = batch_size

    def predict_ensemble(self):
        ensemble_predictions = []

        # Use the same test data for each model
        test_X, test_ac, _, test_y = self.test_data_loader.get_batch(self.batch_size, self.device)
        test_X, test_y = test_X.to(self.device), test_y.to(self.device)

        for model in self.models:
            model.to(self.device)
            model.eval()

            with torch.no_grad():
                model_predictions = model(test_X, test_ac)
                ensemble_predictions.append(model_predictions.cpu().numpy())

        # Calculate the aggregated predictions
        print(np.array(ensemble_predictions).shape)
        aggregated_predictions = np.mean(ensemble_predictions, axis=0)
        
        print(aggregated_predictions.shape)

        # Convert to torch tensors
        aggregated_predictions_tensor = torch.tensor(aggregated_predictions, device=self.device)
        test_y = torch.tensor(test_y, device=self.device)

        # Calculate SmoothL1 loss
        ensemble_loss = nn.SmoothL1Loss()(aggregated_predictions_tensor, test_y)

        return aggregated_predictions, ensemble_loss.item()
    
    def save_ensemble_model(self, save_path):
        # Save the state dictionaries of individual models
        ensemble_state_dicts = [model.state_dict() for model in self.models]
        torch.save(ensemble_state_dicts, save_path)
        print(f"Ensemble model saved to: {save_path}")

def check():
    print('Ayub')