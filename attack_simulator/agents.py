from abc import ABC, abstractmethod
import numpy as np
from numpy.core.fromnumeric import argmax
import numpy.random as random

class Agent(ABC):

	@abstractmethod
	def act(self, observation):
		...

	@abstractmethod
	def update(self, action, reward):
		...

class BanditAgent(Agent):

	def __init__(self, num_actions) -> None:
		self.Q = np.array(num_actions)
		self.times_taken = np.array(num_actions)

	def update(self, action, reward):
		self.times_taken[action] += 1	
		self.Q[action] = self.Q[action] + (reward-self.Q[action])/self.times_taken[action]

	def act(self, observation):
		return np.argmax(self.Q)

class QLearningAgent(Agent):

	def __init__(self, num_actions) -> None:
		
		pass

	def update(self):
		pass

class RandomAgent(Agent):

	def __init__(self, num_services, disable_probability):
		self.disable_probability = disable_probability
		self.num_services = num_services

	def act(self, enabled_services):
		for service in enabled_services:
			if enabled_services[service] == 1 and random.uniform(0,1) < self.disable_probability:
				enabled_services[service] = 0
		return enabled_services

	def update(self, action, reward):
		return
