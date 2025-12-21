import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, in_dim=3, hidden=(64, 64), out_dim=1, dropout=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    


if __name__ == "__main__":
    nn_structure = (16, 32, 64, 64, 32, 16)
    model = MLPRegressor(in_dim=3, hidden=nn_structure, out_dim=1, dropout=0.1)
    x = torch.randn(10, 3)
    y = model(x)
    print(y.shape)
    print('done')