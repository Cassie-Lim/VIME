import gym
import torch
import numpy as np
import torch.nn as nn
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import ValueFunctionQ, Policy
from src.buffer import ReplayBuffer, Transition

DEVICE = device()


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


# Hint: loss you can use to update Q function
def loss_Q(
        value_batch: torch.Tensor, target_batch: torch.Tensor
) -> torch.Tensor:
    mse = nn.MSELoss()
    return mse(value_batch, target_batch)


# Hint: loss you can use to update policy
def loss_pi(
        log_probabilities: torch.Tensor, advantages: torch.Tensor
) -> torch.Tensor:
    return -1.0 * (log_probabilities * advantages).mean()

# Hint: you can use similar implementation from dqn algorithm
def optimize_Q(
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        policy: Policy,
        gamma: float,
        batch: Transition,
        optimizer: Optimizer
):
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

    # actions_, log_probabilities = policy.sample_multiple(states)
    # actions_ = actions_.unsqueeze(-1)[nonterminal_mask]

    rewards = tensor(rewards)
    # TODO: Update the Q-network

    # # calculate the target
    # targets = torch.zeros(size=(batch_size, 1), device=DEVICE)
    # with torch.no_grad():
    #     # Hint: Compute the target Q-values
        
    targets = rewards
    with torch.no_grad():
        # Students are expected to compute the target Q-values here
        targets[nonterminal_mask] += gamma * target_Q.forward(valid_next_states.astype(np.float32)).max(dim=-1)[0]
    
    values = Q(states, actions)
    # print("opt q", values.shape, targets.shape)
    loss_q = loss_Q(values, targets)

    optimizer.zero_grad()
    loss_q.backward()
    optimizer.step()






# Hint: you can use similar implementation from reinforce algorithm
def optimize_policy(
        policy: Policy,
        Q: ValueFunctionQ,
        batch: Transition,
        optimizer: Optimizer
):
    states = np.stack(batch.state)

    actions, log_probabilities = policy.sample_multiple(states)


    # TODO: Update the policy network

    with torch.no_grad():
        # Hint: Advantages
        advantages = Q(states, actions) - Q.forward(states).max(dim=-1)[0]

    # print("opt policy", log_probabilities.shape, advantages.shape)
    l = loss_pi(log_probabilities, advantages)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()



def train_one_epoch(
        env: gym.Env,
        policy: Policy,
        Q: ValueFunctionQ,
        target_Q: ValueFunctionQ,
        memory: ReplayBuffer,
        optimizer_Q: Optimizer,
        optimizer_pi: Optimizer,
        gamma: float = 0.99,
) -> float:
    # make sure target isn't fitted
    policy.train(), Q.train(), target_Q.eval()

    # Reset the environment and get a fresh observation
    state, info = env.reset()

    for t in count():

        # TODO: Complete the train_one_epoch for actor-critic algorithm
        action, _ = policy.sample(state)
        next_state, reward, terminated, trunctuated, _ = env.step(action)
        done = terminated or trunctuated

        if done:
            next_state = None

        # TODO: Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Hint: Use replay buffer!
        # Hint: Check if replay buffer has enough samples
        if len(memory) >= memory.batch_size:
            batch_transitions = memory.sample()
            batch = Transition(*zip(*batch_transitions))
            optimize_Q(Q, target_Q, policy, gamma, batch, optimizer_Q)
            optimize_policy(policy, Q, batch, optimizer_pi)
        if done:
            break


    # Placeholder return value (to be replaced with actual calculation)
    return 0.0
