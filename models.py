import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.network(x)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)

class DDPGActor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
        self.temperature = 0.5

    def forward(self, state):
        logits = self.net(state)
        scaled_logits = logits / self.temperature
        weights = F.softmax(scaled_logits, dim=-1)
        weights = torch.clamp(weights, 0.0, 0.5)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights

    def act(self, state, noise_scale=0.1, explore=True):
        weights = self.forward(state)
        if explore and noise_scale > 0:
            weights = weights + noise_scale * torch.randn_like(weights)
            weights = torch.clamp(weights, min=1e-8)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return weights

class DDPGCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256,128), init_w=0.01):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.final = nn.Linear(in_dim, action_dim)
        self.final.weight.data.uniform_(-init_w, init_w)
        self.final.bias.data.fill_(0)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        logits = self.final(x)
        return logits

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256,128)):
        super().__init__()
        layers1 = []
        in_dim = state_dim + action_dim
        for h in hidden:
            layers1 += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers1.append(nn.Linear(in_dim, 1))
        self.q1 = nn.Sequential(*layers1)

        layers2 = []
        in_dim = state_dim + action_dim
        for h in hidden:
            layers2 += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers2.append(nn.Linear(in_dim, 1))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 128), init_w=0.01):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.network = nn.Sequential(*layers)
        self.final = nn.Linear(in_dim, action_dim)
        self.final.weight.data.uniform_(-init_w, init_w)
        self.final.bias.data.fill_(0)

    def forward(self, x):
        x = self.network(x)
        logits = self.final(x)
        return logits

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden=(256, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
