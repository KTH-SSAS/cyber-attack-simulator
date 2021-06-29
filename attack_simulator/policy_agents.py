import torch.nn as nn
from torch.nn.modules.activation import Softmax
from torch.distributions import Categorical
from torch import Tensor
from torch.optim import optimizer
from attack_simulator.tabular_agents import Agent
import torch

import numpy as np


# design influenced by https://github.com/pytorch/examples/tree/master/reinforcement_learning
class Reinforce(nn.Module):

	def __init__(self, input_dim, num_actions, hidden_dim):
		super().__init__()

		self.model = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(hidden_dim, num_actions),
			nn.Softmax(dim=0)
		)



	def forward(self, state):
		return self.model(state)

class ReinforceAgent(Agent):

	def __init__(self, input_dim, num_actions, hidden_dim, gamma=0.1) -> None:
		self.policy = Reinforce(input_dim, num_actions, hidden_dim)
		self.saved_log_probs = []
		self.gamma = gamma
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)

	def act(self, state):
		action_probabilties = self.policy.forward(torch.Tensor(state))
		# Create a distribution from the action probabilities...
		m = Categorical(action_probabilties)
		action: Tensor = m.sample()  # and sample an action from it.
		self.saved_log_probs.append(m.log_prob(action))
		return action.item()

	def update(self, rewards):
		eps = np.finfo(np.float32).eps.item() # for numerical stability
		R = 0 # Return at t=0
		returns = torch.zeros((len(rewards))) # Array to store returns
		loss = torch.zeros((len(rewards)))
		for i, reward in enumerate(rewards):
			R = reward + self.gamma * R
			returns[i] = R
		
		returns = torch.Tensor(returns)

		#if len(returns) > 1:
		#	returns = (returns-returns.mean())/(returns.std()+eps) # normalize returns
		#else:
		#	returns = (returns-returns.mean())/(eps) # normalize returns

		for i, (log_prob, R) in enumerate(zip(self.saved_log_probs, returns)):
			loss[i] = -log_prob * R # minus sign on loss for gradient ascent

		loss = loss.sum()
		loss.backward()
		torch.nn.utils.clip_grad_value_(self.policy.parameters(), 1)
		self.optimizer.step()
		self.saved_log_probs = []
		self.rewards = []
		return loss.item()
