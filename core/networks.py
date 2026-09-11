import torch
import torch.nn as nn
from torch.distributions import Normal


def build_mlp(input_dim, output_dim, hidden_sizes=(64, 64)):
    layers = []
    last_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.Tanh())
        last_dim = hidden_dim

    layers.append(nn.Linear(last_dim, output_dim))

    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()

        self.mean_network = build_mlp(
            obs_dim,
            act_dim,
            hidden_sizes
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.mean_network(obs)
        std = torch.exp(self.log_std)

        return Normal(mean, std)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()

        self.value_network = build_mlp(
            obs_dim,
            1,
            hidden_sizes
        )

    def forward(self, obs):
        return self.value_network(obs).squeeze(-1)