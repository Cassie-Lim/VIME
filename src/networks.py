import torch
import numpy as np
from torch import nn
import torch.functional as F
from typing import Tuple, Optional
from torch.distributions.categorical import Categorical

from src.utils import device
from src.vime import VIME

DEVICE = device()
HIDDEN_DIMENSION: int = 64


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)

def weights_init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def network(
        in_dimension: int, hidden_dimension: int, out_dimension: int, hidden_layers: int = 1
) -> nn.Module:
    """
    TODO: Design and implement the neural network architecture.

    Args:
        in_dimension (int): Dimension of the input layer.
        hidden_dimension (int): Dimension of the hidden layers.
        out_dimension (int): Dimension of the output layer.

    Returns:
        nn.Module: The constructed neural network model.
    """
    layers = []
    layers.append(nn.Linear(in_dimension, hidden_dimension))
    layers.append(nn.ReLU())
    for _ in range(hidden_layers):
        layers.append(nn.Linear(hidden_dimension, hidden_dimension))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dimension, out_dimension))
    # layers.append(nn.Softmax(dim=-1))
    model = nn.Sequential(*layers)
    # model.apply(weights_init_he)
    return model


class Policy(nn.Module):
    def __init__(
            self,
            state_dimension: int,
            num_actions: int,
            hidden_dimension: int = HIDDEN_DIMENSION,
            hidden_layers: int = 1,
            with_vime: bool = True,
            beta=0.1,
            vime_lr=1e-3
    ):
        super(Policy, self).__init__()
        self.network = network(
            state_dimension, hidden_dimension, num_actions, hidden_layers
        )
        if with_vime:
            self.dynamics_model = VIME(state_dimension, 1, beta=beta, lr=vime_lr)
            self.dynamics_model.dynamics_model.to(DEVICE)  # Initialize VIME


    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the Policy network.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The output logits for each action.

        TODO: Implement the forward method to compute the network output for the given state.
        You can use the self.network to forward the input.
        """
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        logits = self.network(torch.from_numpy(state).to(DEVICE))
        return logits


    def pi(self, state: np.ndarray) -> Categorical:
        """
        Computes the action distribution π(a|s) for a given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Categorical: The action distribution.

        TODO: Implement the pi method to create a Categorical distribution based on the network's output.
        """
        logits = self.forward(state)
        # return Categorical(probs=logits)

        return Categorical(logits=logits)
        # probabilities = torch.softmax(logits, dim=-1)
        # return Categorical(probs=probabilities)


    def sample(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Samples an action from the policy and returns the action along with its log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Tuple[int, torch.Tensor]: The sampled action and its log probability.

        TODO: Implement the sample method to sample an action and compute its log probability.
        """
        action_distribution = self.pi(state)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        return action.item(), log_prob

    def sample_multiple(self, states: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Samples actions for multiple states and returns the actions along with their log probabilities.

        Args:
            states (np.ndarray): The input states.

        Returns:
            Tuple[int, torch.Tensor]: The sampled actions and their log probabilities.

        TODO: Implement the sample_multiple method to handle multiple states.
        """
        action_distribution = self.pi(states)
        actions = action_distribution.sample()
        log_probs = action_distribution.log_prob(actions)
        return actions, log_probs


    def action(self, state: np.ndarray) -> torch.Tensor:
        """
        Selects an action based on the policy without returning the log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The selected action.

        TODO: Implement the action method to return an action based on the sampled action.
        """
        action, _ = self.sample(state)
        return action


class ValueFunctionQ(nn.Module):
    def __init__(
            self,
            state_dimension: int,
            num_actions: int,
            hidden_dimension: int = HIDDEN_DIMENSION,
            with_vime: bool = True,
            beta=0.1,
            vime_lr=3e-4
    ):
        super(ValueFunctionQ, self).__init__()
        self.num_actions = num_actions
        self.network = network(
            state_dimension, hidden_dimension, num_actions
        )
        if with_vime:
            self.dynamics_model = VIME(state_dimension, 1, beta=beta, lr=vime_lr)
            self.dynamics_model.dynamics_model.to(DEVICE)  # Initialize VIME

    def __call__(
            self, state: np.ndarray, action: Optional[int] = None
    ) -> torch.Tensor:
        """
        Computes the Q-values Q(s, a) for given states and optionally for specific actions.

        Args:
            state (np.ndarray): The input state.
            action (Optional[int], optional): The action for which to compute the Q-value. Defaults to None.

        Returns:
            torch.Tensor: The Q-values.

        TODO: Implement the __call__ method to return Q-values for the given state and action.
        This method is intended to compute Q(s, a).
        """
        q_fn = self.forward(state)
        if action is not None:
            if len(q_fn.shape) == 1:
                return q_fn[action]
            action = torch.as_tensor(action, dtype=torch.int64, device=DEVICE)
            q_values = q_fn.gather(1, action.unsqueeze(-1))
            return q_values.squeeze(-1)
        else:
            return q_fn
        

    def forward(self, state: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the ValueFunctionQ network.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The Q-values for each action.

        TODO: Implement the forward method to compute Q-values for the given state.
        You can use the self.network to forward the input.
        """
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return self.network(torch.from_numpy(state).to(DEVICE)).squeeze()

    def greedy(self, state: np.ndarray) -> torch.Tensor:
        """
        Selects the action with the highest Q-value for a given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The action with the highest Q-value.

        TODO: Implement the greedy method to select the best action based on Q-values.
        This method is intended for greedy sampling.
        """
        q_fn = self.forward(state)
        return torch.argmax(q_fn, dim=-1)

    def action(self, state: np.ndarray) -> np.ndarray:
    # def action(self, state: np.ndarray) -> torch.Tensor:
        """
        Returns the greedy action for the given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The selected action.

        TODO: Implement the action method to return the greedy action.
        """
        return self.greedy(state).cpu().numpy()

    def V(self, state: np.ndarray, policy: Policy) -> float:
        """
        Computes the expected value V(s) of the state under the given policy.

        Args:
            state (np.ndarray): The input state.
            policy (Policy): The policy to evaluate.

        Returns:
            float: The expected value.

        TODO: Implement the V method to compute the expected value of the state under the policy.
        This method is intended to return V(s).
        """
        action_distribution = policy.pi(state)
        q_fn = self.forward(state)
        return (action_distribution.probs * q_fn).sum()
