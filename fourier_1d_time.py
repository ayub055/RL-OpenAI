"""
This file is the Fourier Neural Operator for 1D problem,
which uses a recurrent structure to propagate in time.
"""


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ArgStorage:
    def __init__(self, args: dict) -> None:
        self.__dict__.update(args)


################################################################
# fourier layer
################################################################

class SpectralConv1d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d_fast, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] =\
            self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, history):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 1 location (u(t-10, x), ..., u(t-1, x),  x)
        input shape: (batchsize, x=64, c=11)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, c=1)
        """

        self.modes = modes
        self.width = width
        self.history = history
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(self.history + 1, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d_fast(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batch_size, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batch_size, 1, 1])
        return gridx.to(device)


################################################################
# data
################################################################


def get_data(path, history):
    df = pd.read_csv(path)
    df = df.dropna()

    data = df.values[:, 1:4]
    data2 = np.zeros_like(data)
    for i in range(1, len(data)):
        data2[i] = data[i] - data[i-1]
    data2[data2 < 0] = 0
    data = data2

    # split data into train and test
    train_size = int(len(data) * 0.8)
    train_set, test_set = data[:train_size, :], data[train_size:, :]

    # get each datapoint into the shape of x = (batch_size, x=8, c=10), y = (batch_size, x=8, c=1)
    history = history + 1
    
    train_data = []
    for i in range(history, len(train_set)):
        train_data.append(train_set[i-history:i, :])
    train_data = np.array(train_data, dtype=np.float32)
    train_data = torch.from_numpy(train_data).float()
    train_data = train_data.permute(0, 2, 1)

    test_data = []
    for i in range(history, len(test_set)):
        test_data.append(test_set[i-history:i, :])
    test_data = np.array(test_data, dtype=np.float32)
    test_data = torch.from_numpy(test_data).float()
    test_data = test_data.permute(0, 2, 1)

    return train_data, test_data


def get_batch(dataset, batch_size):
    idxs = np.random.choice(len(dataset), batch_size, replace=False)
    batch = dataset[idxs]
    batch_x = batch[:, :, :-1]
    batch_y = batch[:, :, -1:]
    return batch_x, batch_y


################################################################
# training and evaluation
################################################################

if __name__ == '__main__':
    args = ArgStorage({
        'seed' : 2023,
        'gpu_id' : 0,
        'modes' : 2, # 16,
        'width' : 64,
        'epochs' : 1000,
        'history' : 28, # 10,
        'batch_size' : 256,
        'learning_rate' : 0.0001,
        'weight_decay' : 1e-5,
        'path' : './COVID19_data.csv',
    })

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = FNO1d(args.modes, args.width, args.history).to(device)
    # inp = torch.randn(4, 32, 10).cuda() # given 10 time steps (32 dimensional input)
    # out = model(inp)
    # print(out.shape) # torch.Size([4, 32, 1]) # predict next time step
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    train_data, test_data = get_data(args.path, args.history)
    test_x = test_data[:, :, :-1]
    test_y = test_data[:, :, -1:]
    test_x, test_y = test_x.to(device), test_y.to(device)

    # # train
    # hist = np.inf
    # for i in range(5_000):
    #     x, y = get_batch(train_data, args.batch_size)
    #     x, y = x.to(device), y.to(device)

    #     y_hat = model(x)
    #     loss = F.smooth_l1_loss(y_hat, y)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if i % 100 == 0:
    #         with torch.no_grad():
    #             test_y_hat = model(test_x)
    #         error = F.smooth_l1_loss(test_y_hat, test_y)
    #         print(i, ", Train Error:", loss.item(), ", Test Error:", error.item())

    #         if error.item() < hist:
    #             torch.save(model.state_dict(), 'model.pth')
    #             hist = error.item()

    model.load_state_dict(torch.load('model.pth'))
    
    # evaluate
    test_y_hat = model(test_x)
    loss = F.smooth_l1_loss(test_y_hat, test_y)
    print("Test Error:", loss.item())

    idx = 0
    ys = list(train_data[:, idx, -1]) + test_x[:, idx, 0].tolist() + test_y[:, idx, :].squeeze().tolist()
    plt.plot(ys, label='Ground Truth')
    xs = np.arange(len(ys) - len(test_y_hat[:, idx, :]), len(ys))
    plt.plot(xs, test_y_hat[:, idx, :].detach().cpu().numpy(), label='Prediction')
    plt.legend()
    plt.grid()
    plt.savefig('plot_cases.png')
    plt.close()

    idx = 1
    ys = list(train_data[:, idx, -1]) + test_x[:, idx, 0].tolist() + test_y[:, idx, :].squeeze().tolist()
    plt.plot(ys, label='Ground Truth')
    xs = np.arange(len(ys) - len(test_y_hat[:, idx, :]), len(ys))
    plt.plot(xs, test_y_hat[:, idx, :].detach().cpu().numpy(), label='Prediction')
    plt.legend()
    plt.grid()
    plt.savefig('plot_hospitalizations.png')
    plt.close()

    idx = 2
    ys = list(train_data[:, idx, -1]) + test_x[:, idx, 0].tolist() + test_y[:, idx, :].squeeze().tolist()
    plt.plot(ys, label='Ground Truth')
    xs = np.arange(len(ys) - len(test_y_hat[:, idx, :]), len(ys))
    plt.plot(xs, test_y_hat[:, idx, :].detach().cpu().numpy(), label='Prediction')
    plt.legend()
    plt.grid()
    plt.savefig('plot_deaths.png')
    plt.close()
