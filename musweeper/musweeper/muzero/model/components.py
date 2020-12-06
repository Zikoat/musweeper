import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules import transformer
from ..utils.debugger import *
import numpy as np

def transform_input(tensor):
	if isinstance(tensor, np.ndarray):
		tensor = torch.from_numpy(tensor)
		tensor = torch.flatten(tensor) if tensor.dim() != 1 else tensor
		tensor = tensor.float()
	
	if not tensor.is_cuda and torch.cuda.is_available():
		return tensor.cuda()
	return tensor

class shared_backbone:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.debugger = model_debugger()

class component_dynamics(nn.Module, shared_backbone):
	"""
	Description of the dynamics component of Muzero 
	the role of the dynamics component is to learn the effect of actions
	in diffrent states.
	"""
	def __init__(self, representation_size):
		super(component_dynamics, self).__init__()
		shared_backbone.__init__(self)
		self.shared_hidden_size = 64

		self.preprocess_state = nn.Linear(representation_size, self.shared_hidden_size).to(self.device)
		self.preprocess_action = nn.Linear(1, self.shared_hidden_size).to(self.device)

		self.combined = nn.Linear(2 * self.shared_hidden_size, 128).to(self.device)

		self.reward = nn.Linear(128, 1).to(self.device)
		self.new_state = nn.Linear(128, representation_size).to(self.device)

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
		if len(state.shape) == 1:
			state = state.reshape((1, -1))

		if len(action.shape) == 1:
			action = action.reshape((1, -1))

		state = transform_input(state)
		action = transform_input(action)

		state = self.preprocess_state(state)
		state = torch.sigmoid(state.view(state.size(0), -1))
		action = self.preprocess_action(action)
		action = torch.sigmoid(action.view(action.size(0), -1))

		combined = torch.sigmoid(self.combined(torch.cat((state, action), dim=1)))
		return torch.sigmoid(self.new_state(combined)), torch.sigmoid(self.reward(combined))


class component_predictions(nn.Module, shared_backbone):
	"""
	The prediction component will learn the optimal state and value function
	this is predicted from the current state from the representation component.
	"""
	def __init__(self, representation_size, action_size):
		super(component_predictions, self).__init__()
		shared_backbone.__init__(self)

		main_hidden_layer_size = 128
		second_hidden_layer_size = main_hidden_layer_size // 2

		self.preprocess_state = nn.Linear(representation_size, main_hidden_layer_size).to(self.device)
		self.combined = nn.Linear(main_hidden_layer_size, second_hidden_layer_size).to(self.device)

		self.value = nn.Linear(second_hidden_layer_size, 1).to(self.device)
		self.policy = nn.Linear(second_hidden_layer_size, action_size).to(self.device)

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
		self.debugger.start_forward("predictions")
		state = transform_input(state)
		self.debugger.add_element("predictions", state)
		state = torch.sigmoid(self.preprocess_state(state))
		self.debugger.add_element("predictions", state)
		combined = torch.sigmoid(self.combined(state))
		self.debugger.add_element("predictions", combined)
		policy, value = torch.softmax(torch.sigmoid(self.policy(combined)), dim=1), torch.sigmoid(self.value(combined))
		self.debugger.add_element("predictions", (policy, value))
		self.debugger.stop_forward("predictions")
		return policy, value

class component_representation(nn.Module, shared_backbone):
	"""
	The representation component will convert the real environment state to lantent space
	muzero does not use environment state directly. It converts it to latent space and
	uses that in the other components.
	"""
	def __init__(self, env_size, representation_size):
		super(component_representation, self).__init__()
		shared_backbone.__init__(self)

		main_hidden_layer_size = 128
		second_hidden_layer_size = main_hidden_layer_size // 2

		self.preprocess_state = nn.Linear(env_size, main_hidden_layer_size).to(self.device)
		self.combined = nn.Linear(main_hidden_layer_size, second_hidden_layer_size).to(self.device)
		self.representation = nn.Linear(second_hidden_layer_size, representation_size).to(self.device)

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
		state = transform_input(state)
		state = torch.sigmoid(self.preprocess_state(state))
		combined = torch.sigmoid(self.combined(state))

		return torch.sigmoid(self.representation(combined))

