import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gym
from vime import VIME
# PPO Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, is_discrete=False):
        super(PolicyNetwork, self).__init__()
        self.is_discrete = is_discrete
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        if is_discrete:
            self.logits = nn.Linear(hidden_dim, action_dim)  # Logits for discrete actions
        else:
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.fc(state)
        if self.is_discrete:
            return self.logits(x)
        else:
            return self.mean(x), self.log_std.exp()

    def act(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if self.is_discrete:
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action)
        else:
            mean, std = self.forward(state)
            dist = Normal(mean, std)
            action = dist.sample()
            return action, dist.log_prob(action).sum(dim=-1)



class PPOAgent:
    def __init__(self, state_dim, action_dim, is_discrete, beta=0.1, lr=3e-4, gamma=0.99, clip_epsilon=0.2, K_epochs=10):
        self.policy = PolicyNetwork(state_dim, action_dim, is_discrete=is_discrete)
        self.is_discrete = is_discrete
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs
        dynamic_action_dim = 1 if is_discrete else action_dim
        self.dynamics_model = VIME(state_dim, dynamic_action_dim, beta=beta, lr=lr)  # Initialize VIME

    def compute_returns(self, rewards, masks):
        returns = []
        R = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            R = reward + self.gamma * R * mask
            returns.insert(0, R)
        return torch.tensor(returns)

    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.K_epochs):
            if self.is_discrete:
                logits = self.policy(states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
            else:
                mean, std = self.policy(states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = (log_probs - log_probs_old).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean()  # PPO objective

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, env, episodes=500, max_timesteps=300):
        for episode in range(episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            states, actions, log_probs, rewards, masks = [], [], [], [], []
            intrinsic_rewards = []
            for t in range(max_timesteps):
                action, log_prob = self.policy.act(state)
                
                # For discrete actions, we need to pass an integer, for continuous, a tensor
                action_input = action.item() if self.is_discrete else action.numpy()
                
                next_state, reward, terminated, truncated, _ = env.step(action_input)
                done = terminated or truncated

                # Calculate intrinsic reward with VIME
                next_state = torch.tensor(next_state, dtype=torch.float32)
                intrinsic_reward = self.dynamics_model.intrinsic_reward(state, action, next_state).item()
                total_reward = reward + intrinsic_reward  # Combine rewards

                # Store transitions
                states.append(state.detach())
                actions.append(action.detach())
                log_probs.append(log_prob.detach())
                rewards.append(total_reward)
                masks.append(1 - done)

                # Move to next state
                state = next_state
                if done:
                    break

            # Process trajectory
            states = torch.stack(states)
            actions = torch.stack(actions)
            log_probs_old = torch.stack(log_probs)
            returns = self.compute_returns(rewards, masks)
            advantages = returns - returns.mean()  # Centered advantages for PPO

            # Update policy using PPO
            self.update(states, actions, log_probs_old, returns, advantages)

            # Logging
            print(f"Episode {episode}, Reward: {sum(rewards):.2f}")

if __name__ == "__main__":
    env = gym.make("CartPole-v1")  # Change to continuous env if needed
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

    agent = PPOAgent(state_dim, action_dim, is_discrete=is_discrete)
    agent.train(env)  # Start training the agent
    torch.save(agent.policy.state_dict(), "ppo_policy.pth")