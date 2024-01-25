import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, state_dim, ac_dim, hid_dim=512) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim+ac_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3s = nn.Linear(hid_dim, state_dim)
        self.fc3r = nn.Linear(hid_dim, 1)
    
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        d = self.fc3s(x)
        r = self.fc3r(x)
        return d+s, r


def get_dynamics_ensemble(
    ensemble_size, env, hid_dim, device, lr, wd) -> list:
    state_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    models = [
        MLP(state_dim, ac_dim, hid_dim).to(device) for _ in range(ensemble_size)
    ]
    optimizers = [
        optim.Adam(model.parameters(), lr=lr, weight_decay=wd)\
            for model in models]
    return models, optimizers


class Policy(nn.Module):
    def __init__(self, state_dim, ac_dim, hid_dim) -> None:
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, ac_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)