import gym
import torch
import numpy as np
import torch.nn as nn
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ
from src.buffer import ReplayBuffer, Transition

DEVICE = device()
EPS_END: float = 0.01
EPS_START: float = 1.0
EPS_DECAY: float = 0.999_9
eps: float = EPS_START


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


# simple MSE loss
# Hint: used for optimize Q function
def loss(
        value_batch: torch.Tensor, target_batch: torch.Tensor
) -> torch.Tensor:
    mse = nn.MSELoss()
    return mse(value_batch, target_batch)


def greedy_sample(Q: ValueFunctionQ, state):
    with torch.no_grad():
        return Q.action(state)


def eps_greedy_sample(Q: ValueFunctionQ, state: np.array):
    global eps
    eps = max(EPS_END, EPS_DECAY * eps)

    # TODO: Implement epsilon-greedy action selection
    # You can copy from your previous implementation
    # With probability eps, select a random action
    # With probability (1 - eps), select the best action using greedy_sample
    if np.random.rand() < eps:
        return torch.randint(0, Q.num_actions, (1,))
    else:
        return greedy_sample(Q, state).reshape((1,))


def optimize_Q(
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        gamma: float,
        memory: ReplayBuffer,
        optimizer: Optimizer
):
    if len(memory) < memory.batch_size:
        return

    batch_transitions = memory.sample()
    batch = Transition(*zip(*batch_transitions))

    states = np.stack(batch.state)
    actions = np.stack(batch.action)
    rewards = np.stack(batch.reward)
    valid_next_states = np.stack(tuple(
        filter(lambda s: s is not None, batch.next_state)
    ))

    nonterminal_mask = tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        type=torch.bool
    )

    rewards = tensor(rewards)

    # TODO: Update the Q-network
    # Hint: Calculate the target Q-values
    # Initialize targets with zeros
    # targets = torch.zeros(size=(memory.batch_size, 1), device=DEVICE)
    targets = rewards
    with torch.no_grad():
        # Students are expected to compute the target Q-values here
        targets[nonterminal_mask] += gamma * target_Q.forward(valid_next_states.astype(np.float32)).max(dim=-1)[0].detach()
    values = Q(states, actions)
    # print(values.requires_grad, targets.requires_grad, values.device, targets.device)
    l = loss(values.squeeze(), targets.squeeze())

    optimizer.zero_grad()
    l.backward()
    optimizer.step()



def train_one_epoch(
        env: gym.Env,
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        memory: ReplayBuffer,
        optimizer: Optimizer,
        gamma: float = 0.99
) -> float:
    # Make sure target isn't being trained
    Q.train()
    target_Q.eval()

    # Reset the environment and get a fresh observation
    state, info = env.reset()

    episode_reward: float = 0.0

    for t in count():
        # TODO: Complete the train_one_epoch for dqn algorithm
        action = eps_greedy_sample(Q, state).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            next_state = None

        episode_reward += reward
        memory.push(state, action, next_state, reward)  # ('state', 'action', 'next_state', 'reward')

        state = next_state

        # Optimize Q-network if the replay buffer is sufficiently full
        if len(memory) >= memory.batch_size:
            optimize_Q(Q, target_Q, gamma, memory, optimizer)
        if done:
            break


    # Placeholder return value (to be replaced with actual calculation)
    return episode_reward
