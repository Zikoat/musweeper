import gym
import torch


class BasicEnv(gym.Env):	
	"""
	Basic env for testing model. 
	The model should be able to learn the state changees.
	"""
	def __init__(self, state_size=2):
		self.round = 0
		self.state_size = state_size
		self.action_size = self.state_size
		self.timeout = 10

	@property
	def active_index(self):
		return int(self.round % self.state_size == 0)

	@property
	def state(self):
		"""
		Returns the state representation

		Returns
		-------
		torch.tensor
			the output state
		"""
		state = torch.zeros((self.state_size))
		state[self.active_index] = 1
		return state

	def reset(self):
		"""Reset the current environment

		Returns
		-------
		torch.tensor
			The start state
		"""
		self.round = 0
		return self.state

	def step(self, action):
		"""
		Do a step in the environment

		Parameters
		----------
		action : int
			the action index

		Returns
		-------
		torch.tensor
			the new state
		int
			the reward
		boolean
			if the game is done
		"""
		reward = int(action == self.active_index)

		self.round += 1
		return self.state, reward, (self.round > self.timeout)

	def value(self, action, t):
		"""
		Value function of each state

		Parameters
		----------
		action : int
			the taken taken
		t : int
			timestemp

		Returns
		-------
		int
			reward -1 for bad action and 1 for good action
		"""
		return 1 if bool(action == self.active_index) else -1
