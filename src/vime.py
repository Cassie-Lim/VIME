import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Dynamics model for predicting next state
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(DynamicsModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)  # Mean and log variance
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)  # Output includes mean and log variance

# Intrinsic reward based on information gain
class VIME(nn.Module):
    def __init__(self, state_dim, action_dim, beta=0.1, lr=1e-3):
        super(VIME, self).__init__()
        self.dynamics_model = DynamicsModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.dynamics_model.parameters(), lr=lr)
        self.beta = beta  # Scaling factor for intrinsic reward

    def predict(self, state, action):
        mean_logvar = self.dynamics_model(state, action)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return Normal(mean, std)

    def intrinsic_reward(self, state, action, next_state):
        prior_dist = self.predict(state, action)
        self.update_model(state, action, next_state)
        posterior_dist = self.predict(state, action)
        kl_divergence = torch.distributions.kl_divergence(posterior_dist, prior_dist).sum(dim=-1)
        # print("KL Divergence:", kl_divergence.shape, kl_divergence.mean())
        return self.beta * kl_divergence

    def update_model(self, state, action, next_state):
        pred_dist = self.predict(state, action)
        log_prob = pred_dist.log_prob(next_state).sum(dim=-1)
        loss = -log_prob.mean()
        self.optimizer.zero_grad()
        loss.backward()
        # print(self.dynamics_model.fc[0].weight.grad.mean())
        # print("Dynamic Model Loss:", loss.item())
        self.optimizer.step()
