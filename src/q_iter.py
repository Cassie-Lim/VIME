import gym
import torch
import numpy as np
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ


DEVICE = device()
EPS_END: float = 0.01
EPS_START: float = 1.0
EPS_DECAY: float = 0.999_9
eps: float = EPS_START

# simple MSE loss
def loss(
        value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
    mean_square_error = (value - target)**2
    return mean_square_error


def greedy_sample(Q: ValueFunctionQ, state: np.array):
    with torch.no_grad():
        return Q.action(state)


def eps_greedy_sample(Q: ValueFunctionQ, state: np.array):
    global eps
    eps = max(EPS_END, EPS_DECAY * eps)

    # TODO: Implement epsilon-greedy action selection
    # Hint: With probability eps, select a random action
    # Hint: With probability (1 - eps), select the best action using greedy_sample
    if np.random.rand() < eps:
        return torch.randint(0, Q.num_actions, (1,))
    else:
        return greedy_sample(Q, state)
    

def train_one_epoch(
        env: gym.Env,
        Q: ValueFunctionQ,
        optimizer: Optimizer,
        gamma: float = 0.99,
        with_vime: bool = False
    ) -> float:
    Q.train()

    # Reset the environment and get a fresh observation
    state, info = env.reset()

    episode_reward: float = 0.0

    for t in count():
        # TODO: Generate action using epsilon-greedy policy
        action = eps_greedy_sample(Q, state)

        # TODO: Take the action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        if done:
            next_state = None
        orig_reward = reward
        if with_vime:
            intrinsic_reward = Q.dynamics_model.intrinsic_reward(torch.tensor(state).reshape(1, -1).to(DEVICE), torch.tensor(action).reshape(1, -1).to(DEVICE), torch.tensor(next_state).reshape(1, -1).to(DEVICE)).item()
            reward += intrinsic_reward
        # Calculate the target
        with torch.no_grad():
            # TODO: Compute the target Q-value
            if next_state is None:
                target = reward
            else:
                target = reward + gamma * Q.forward(next_state).max().item()

        # TODO: Compute the loss
        value = Q(state, action)
        l = loss(value, torch.tensor(target).to(DEVICE))

        # TODO: Perform backpropagation and update the network
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        # TODO: Update the state
        state = next_state

        # TODO: Handle episode termination
        episode_reward += orig_reward
        if done:
            break

    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
