import gym
import torch
import numpy as np
from typing import Tuple
from itertools import count
from torch.optim import Optimizer

from src.utils import device
from src.networks import Policy


DEVICE = device()


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)

# Hint loss you can use
def loss(
        epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor
    ) -> torch.Tensor:
    return -1.0 * (epoch_log_probability_actions * epoch_action_rewards).mean()

def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    r = 0
    for reward in reversed(rewards):
        r = reward + gamma * r
        discounted_rewards.insert(0, r)
    return discounted_rewards

def train_one_epoch(
        env: gym.Env,
        policy: Policy,
        optimizer: Optimizer,
        max_timesteps=2048,
        with_vime: bool = False
        # max_timesteps=5_000
    ) -> Tuple[float, float]:

    policy.train()

    epoch_total_timesteps = 0

    # Action log probabilities and rewards per step (for calculating loss)
    epoch_log_probability_actions = []
    epoch_action_rewards = []
    

    # Loop through episodes
    while True:
        # Stop if we've done over the total number of timesteps
        if epoch_total_timesteps > max_timesteps:
            break

        # Running total of this episode's rewards
        episode_reward: float = 0
        episode_rewards = []
        states = []
        actions = []
        next_states = []

        # Reset the environment and get a fresh observation
        state, info = env.reset()

        # Loop through timesteps until the episode is done (or the max is hit)
        for t in count():
            epoch_total_timesteps += 1

            # TODO: Sample an action from the policy
            action, log_prob = policy.sample(state)

            # TODO: Take the action in the environment
            next_state, reward, terminated, truncated, info = env.step(action)

            # TODO: Accumulate the reward
            episode_reward += reward
            episode_rewards.append(reward)
            
            # TODO: Store the log probability of the action
            epoch_log_probability_actions.append(log_prob.flatten())
            states.append(state)
            actions.append(action)
            next_states.append(next_state)

            # Finish the action loop if this episode is done
            done = terminated or truncated
            if done:
                # TODO: Assign the episode reward to each timestep in the episode
                
                if with_vime:
                    # Calculate intrinsic reward with VIME
                    shape = (len(states), -1)
                    intrinsic_rewards = policy.dynamics_model.intrinsic_reward(torch.tensor(states).reshape(shape).to(DEVICE), torch.tensor(actions).reshape(shape).to(DEVICE), torch.tensor(next_states).reshape(shape).to(DEVICE)).detach()
                    # print(intrinsic_rewards.sum())
                    episode_rewards = (np.array(episode_rewards) + intrinsic_rewards.cpu().numpy()).tolist()
                    # episode_reward += intrinsic_rewards.sum()
                # epoch_action_rewards += [episode_reward] * (t + 1)
                discounted_rewards = compute_discounted_rewards(episode_rewards)
                epoch_action_rewards += discounted_rewards
                break

    # TODO: Calculate the policy gradient loss
    l = loss(torch.stack(epoch_log_probability_actions).squeeze(), torch.tensor(epoch_action_rewards).to(DEVICE))
    # print(torch.stack(epoch_log_probability_actions).squeeze())
    print(torch.tensor(epoch_action_rewards).to(DEVICE))

    # TODO: Perform backpropagation and update the policy parameters
    optimizer.zero_grad()
    l.backward()
    # print(l, terminated, truncated)
    # print(policy.network[0].weight.grad.mean())
    # print(policy.network[2].weight.grad.mean())
    optimizer.step()


    # Placeholder return values (to be replaced with actual calculations)
    return 0.0, 0.0
