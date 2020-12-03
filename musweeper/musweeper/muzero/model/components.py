import torch.nn as nn
import torch.nn.functional as F
import torch

class component_dynamics(nn.Module):
	"""
	Description of the dynamics component of muzero 
	the role of the dynamics component is to learn the effect of actions
	in diffrent states.
	"""
	def __init__(self, representation_size):
		super(component_dynamics, self).__init__()
		self.shared_hidden_size = 256

		self.preprocess_state = nn.Linear(representation_size, self.shared_hidden_size)
		self.preprocess_action = nn.Linear(1, self.shared_hidden_size)

		self.combined = nn.Linear(2 * self.shared_hidden_size, 128)

		self.reward = nn.Linear(128, 1)
		self.new_state = nn.Linear(128, representation_size)

	def forward(self, state, action):
		"""
		Does a model prediction based on the input	

		Parameters
		----------
		state : torch.tensor
			the current state
		action : tensor
			the action index

		Returns
		-------
		torch.tensor
			the next state transition (from latent space)
		torch.tensor
			the predicted reward from the given action 
		"""
		state = self.preprocess_state(state)
		state = state.view(state.size(0), -1)
		action = self.preprocess_action(action)
		action = action.view(action.size(0), -1)

		combined = self.combined(torch.cat((state, action), dim=1))
		return self.new_state(combined), self.reward(combined)


class component_predictions(nn.Module):
	"""
	The prediction component will learn the optimal state and value function
	this is predicted from the current state from the representation component.
	"""
	def __init__(self, representation_size, action_size):
		super(component_predictions, self).__init__()
		self.preprocess_state = nn.Linear(representation_size, 256)
		self.combined = nn.Linear(256, 128)

		self.value = nn.Linear(128, 1)
		self.policy = nn.Linear(128, action_size)

	def forward(self, state):
		"""
		Does a model prediction based on the input	

		Parameters
		----------
		state : torch.tensor
			the current state

		Returns
		-------
		torch.tensor
			the policy function
		torch.tensor
			the value function
		"""
		if len(state.shape) == 1:
			state = state.reshape((1, -1))
		state = self.preprocess_state(state)
		combined = self.combined(state)

		return self.policy(combined), self.value(combined)

class component_representation(nn.Module):
	"""
	The representation component will convert the real environment state to lantent space
	muzero does not use environment state directly. It converts it to latent space and
	uses that in the other components.
	"""
	def __init__(self, env_size, representation_size):
		super(component_representation, self).__init__()
		self.preprocess_state = nn.Linear(env_size, 256)
		self.combined = nn.Linear(256, 128)
		self.representation = nn.Linear(128, representation_size)

	def forward(self, state):
		"""
		Does a model prediction based on input

		Parameters
		----------
		state : torch.tensor
			the current state from the environment

		Returns
		-------
		torch.tensor
			the state representation in lantent space
		"""
		if len(state.shape) == 1:
			state = state.reshape((1, -1))
		state = self.preprocess_state(state)
		combined = self.combined(state)

		return self.representation(combined)

