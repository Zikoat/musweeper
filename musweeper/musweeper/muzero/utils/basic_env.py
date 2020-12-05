import torch
import gym

class BasicEnv(gym.Env):	
	"""
	Basic env for testing model. 
	The model should be able to learn the state changees.
	"""
	def __init__(self):
		self.round = 0
		self.state_size = 2
		self.action_size = 2
		self.time_out = 10

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
		state[int(self.round % 2 == 0)] = 1
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
		reward = int(action == (self.round % 2 == 0))

		self.round += 1
		return self.state, reward, (self.round > self.time_out)

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
		return 1 if bool(action == (t % 2 == 0)) else -1