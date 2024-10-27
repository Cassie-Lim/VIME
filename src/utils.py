import os
import gym
import torch
import random
import pygame
import numpy as np
import torch.nn as nn
from loguru import logger
from IPython import display
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import animation


ARTIFACT_DIRECTORY: str='artifacts'
MAX_EVAL_EPISODE_STEPS: int = 1_000
EVAL_EPISODES: int = 10
SMOOTHING_WINDOW: int = 50
DEMO_STEPS: int = 1_000
ONLY_CPU: bool = False


def set_seed(seed: int=42):
    random.seed(seed)

    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # uncomment this for better reproducibility; slows torch
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f'Random seed set as {seed}.')


def device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    if ONLY_CPU:
        device = 'cpu'
    
    logger.info(f'Using {device} device.')
    return torch.device(device)


def eval_policy(
        policy: nn.Module,
        environment_name: str='LunarLander-v2'
    ) -> Tuple[float, float]:
    policy.eval()
    
    eval_environment = gym.make(environment_name)
    state, info = eval_environment.reset()

    episode_rewards = []
    with torch.no_grad():
        for episode in range(EVAL_EPISODES):
            episode_reward: float = 0.0
            for step in range(MAX_EVAL_EPISODE_STEPS):
                action = policy.action(state)
                state, reward, terminated, truncated, info = eval_environment.step(action)
                done = terminated or truncated
                episode_reward += reward
                if done:
                    state, info = eval_environment.reset()
                    break
            episode_rewards.append(episode_reward)
    
    eval_environment.close()
    return (
        float(np.mean(episode_rewards)),
        float(np.std(episode_rewards))
    )


def moving_average(data: np.array, window: int=SMOOTHING_WINDOW) -> np.array:
    window = min(window, len(data)//2)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    average = (cumsum[window:] - cumsum[:-window]) / window
    return average


def plot_returns(
        mean_returns: List[float], std_returns: list[float],
        method_name: str, dynamic: bool=False
    ):
    if not os.path.exists(ARTIFACT_DIRECTORY):
        os.makedirs(ARTIFACT_DIRECTORY)
    
    epochs = np.arange(len(mean_returns))
    
    mean_returns = np.array(mean_returns)
    std_returns = np.array(std_returns)

    smoothed = moving_average(mean_returns)

    fig = plt.figure()
    
    plt.plot(epochs, mean_returns, color='b', label='Average Returns')
    plt.plot(
        epochs[-len(smoothed):], smoothed, color='r', label='Smoothed'
    )
    plt.legend(['Average Returns', 'Moving Average'])
    
    low = mean_returns - std_returns
    high = mean_returns + std_returns
    plt.fill_between(epochs, low, high, color='b', alpha = 0.1)

    # some formatting
    plt.ylim(min(mean_returns), max(mean_returns))
    plt.ylabel('Cummulative Rewards')
    plt.xlabel('Epoch')
    plt.title(method_name.split('-')[0])
    plt.tight_layout()
    plt.show()

    plt.pause(0.001)  # pause a bit so that plots are updated
    display.display(plt.gcf())
    if dynamic:
        display.clear_output(wait=True)        
    
    fig.savefig(
        f'{ARTIFACT_DIRECTORY}/{method_name}_returns.png', format='png', dpi=300
    )


def demo_policy(
        policy: nn.Module,
        environment_name: str='LunarLander-v2',
        frame_frequency: int=5
    ) -> List:
    policy.eval()
    demo_environment = gym.make(environment_name, render_mode='rgb_array')
    state, info = demo_environment.reset()
    
    frames = []
    with torch.no_grad():
        for step in range(DEMO_STEPS):
            if not step%frame_frequency:
                frames.append(
                    demo_environment.render()
                )

            action = policy.action(state)
            state, reward, terminated, truncated, info = demo_environment.step(action)
            
            if terminated or truncated:
                frames.append(
                    demo_environment.render()
                )
                state, info = demo_environment.reset()
    
    demo_environment.close()
    if demo_environment.screen is not None:
        pygame.display.quit()
        pygame.quit()
    return frames


def save_frames_as_gif(
        frames: List, method_name: str
    ) -> str:
    if not os.path.exists(ARTIFACT_DIRECTORY):
        os.makedirs(ARTIFACT_DIRECTORY)
    path = f'{ARTIFACT_DIRECTORY}/{method_name}_policy.gif'
    
    # controls frame size
    fig = plt.figure(
        figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0),
        dpi=72
    )

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames = len(frames), interval=50
    )
    anim.save(
        path,
        writer='imagemagick', fps=60
    )
    plt.close(fig)
    return path
