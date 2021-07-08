import torch.nn as nn
from torch.nn.modules.activation import Softmax
from torch.distributions import Categorical
from torch import Tensor
from torch.optim import optimizer
from attack_simulator.tabular_agents import Agent
import torch
import logging
import numpy as np

class StateValueEstimator(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.model(state)

# design influenced by https://github.com/pytorch/examples/tree/master/reinforcement_learning
class Reinforce(nn.Module):

    def __init__(self, input_dim, num_actions, hidden_dim, baseline):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=0)
        )

        if baseline:
            self.baseline = StateValueEstimator(input_dim)
        else:
            self.baseline = None

    def forward(self, state):
        action = self.model(state)
        if self.baseline is None:
            return action
        state_value = self.baseline(state)
        return action, state_value

class ReinforceAgent(Agent):

    def __init__(self, input_dim, num_actions, hidden_dim, gamma=0.9, allow_skip=True, baseline=False) -> None:
        self.reinforce = Reinforce(input_dim, num_actions, hidden_dim, baseline)
        self.saved_log_probs = []
        self.saved_state_values = []
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.reinforce.parameters(), lr=1e-4)
        self.baseline = baseline
        self.can_skip = allow_skip

    def act(self, state):

        state = torch.Tensor(state)

        if self.baseline:
            # Calculate and save the state value in addition to the action
            action_probabilities, state_value = self.reinforce.forward(state)
            self.saved_state_values.append(state_value)
        else:
            action_probabilities = self.reinforce.forward(state)

        # Create a distribution from the action probabilities...
        dist = Categorical(action_probabilities)
        action: Tensor = dist.sample() # sample an action from the distribution
        self.saved_log_probs.append(dist.log_prob(action)) # save the log probability for loss calculation

        return action.item()

    def update(self, rewards):
        loss = self.calculate_loss(rewards)
        loss[0].backward()
        torch.nn.utils.clip_grad_value_(self.reinforce.parameters(), 1)
        self.optimizer.step()
        self.saved_log_probs = []
        self.optimizer.zero_grad()
        return loss[0].item(), loss[1].item(), loss[2].item()

    def calculate_loss(self, rewards, normalize_returns=False):
        eps = np.finfo(np.float32).eps.item()  # for numerical stability
        R = 0  # Return at t=0
        iterations = len(rewards)
        returns = torch.zeros(iterations)  # Array to store returns
        policy_loss = torch.zeros(iterations)

        for i in reversed(range(iterations)):
            R = rewards[i] + self.gamma * R
            returns[i] = R

        returns = torch.Tensor(returns)

        if normalize_returns:
            if len(returns) > 1:
        	    returns = (returns-returns.mean())/(returns.std()+eps) # normalize returns
            else:
        	    returns = returns/returns

        if self.baseline:
            state_values = torch.Tensor(self.saved_state_values)
            returns -= state_values
            value_loss = torch.pow(returns, 2).sum()
            self.saved_state_values = []
            
        for i, (log_prob, R) in enumerate(zip(self.saved_log_probs, returns)):
            policy_loss[i] = -log_prob * R  # minus sign on loss for gradient ascent

        loss = policy_loss.sum()

        if self.baseline:
            loss += value_loss

        if value_loss is None:
            return loss.item()

        return loss, policy_loss.sum(), value_loss


    def eval(self):
        self.reinforce.eval()
    

    def train(self):
        self.reinforce.train()
