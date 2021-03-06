import torch.nn as nn
import torch.nn.functional as F
import torch
from ..utils.debugger import *
import numpy as np

def transform_input(tensor):
	"""
	Transform the input to a general inpt and enables cuda is gpu is avaible

	Parameters
	----------
	tensor: torch.tensor
		the input tensor

	Returns
	-------
	tensor: torch.tensor
		the transfomed tensor
	"""
	if isinstance(tensor, np.ndarray):
		tensor = torch.from_numpy(tensor)
		tensor = tensor.float()

	if not tensor.is_cuda and torch.cuda.is_available():
		return tensor.to(dtype=torch.float32).cuda()
	return tensor.to(dtype=torch.float32)

def convert_to_one_hot(index, action_size):
	"""
	Converts a given input to a one hot vector

	Parameters
	----------
	index: int
		index of the one
	size: int
		size of tensor

	Returns
	-------
	torch.tensor
		output tensor one hot
	"""
	one_hot_action = torch.zeros((1, action_size))
	one_hot_action[0][index] = 1
	return one_hot_action

class shared_backbone:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.debugger = model_debugger()
		self.main_hidden_layer_size = 64

class BasicConv(nn.Module, shared_backbone):
	def __init__(self, output=120):
		super().__init__()
		shared_backbone.__init__(self)
		self.output_size = 96

		self.conv1 = nn.Conv2d(1, 6, 3).to(self.device)
		self.fc1 = nn.Linear(self.output_size, output).to(self.device)

	def forward(self, x):
		"""
		Basic forward operation on conv layer

		Parameters
		----------
		x : torch.tensor
			input data

		Returns
		-------
		torch.tensor
			the output after the operations
		"""
		if len(x.shape) < 4:
			padding = (1, ) * (4 - len(x.shape) )
			x = x.reshape(padding + x.shape)
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = x.view(-1, self.output_size)
		assert x.shape[0] == 1
		assert len(x.shape) == 2
		return F.relu(self.fc1(x))

class component_dynamics(nn.Module, shared_backbone):
	"""
	Description of the dynamics component of Muzero 
	the role of the dynamics component is to learn the effect of actions
	in diffrent states.
	"""
	def __init__(self, representation_size, action_size, config):
		super(component_dynamics, self).__init__()
		shared_backbone.__init__(self)

		main_hidden_layer_size = config.get("main_hidden_layer_size", self.main_hidden_layer_size)
		self.preprocess_state = nn.Linear(representation_size, main_hidden_layer_size).to(self.device)
		self.preprocess_action = nn.Linear(action_size, main_hidden_layer_size).to(self.device)

		self.combined = nn.Linear(2 * main_hidden_layer_size, 128).to(self.device)

		self.reward = nn.Linear(128, 1).to(self.device)
		self.new_state = nn.Linear(128, representation_size).to(self.device)
		self.action_size = action_size

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

		if type(action) == int:
			action = convert_to_one_hot(action, self.action_size)

		if len(action.shape) == 1:
			action = action.reshape((1, -1))

		assert action.shape[-1] == self.action_size, "should be one hot {}".format(action)

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
	def __init__(self, representation_size, action_size, config):
		super(component_predictions, self).__init__()
		shared_backbone.__init__(self)

		main_hidden_layer_size = config.get("main_hidden_layer_size", self.main_hidden_layer_size)
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
		if state.dim() == 1:
			state = state.reshape((1, -1))
		state = transform_input(state)
		state = torch.sigmoid(self.preprocess_state(state))
		combined = torch.sigmoid(self.combined(state))
		policy, value = torch.softmax(torch.sigmoid(self.policy(combined)), dim=1), torch.sigmoid(self.value(combined))
		return policy, value

class component_representation(nn.Module, shared_backbone):
	"""
	The representation component will convert the real environment state to lantent space
	muzero does not use environment state directly. It converts it to latent space and
	uses that in the other components.
	"""
	def __init__(self, env_size, representation_size, config):
		super(component_representation, self).__init__()
		shared_backbone.__init__(self)

		main_hidden_layer_size = config.get("main_hidden_layer_size", self.main_hidden_layer_size)
		second_hidden_layer_size = main_hidden_layer_size // 2

		self.preprocess_state = BasicConv(output=main_hidden_layer_size).to(self.device)
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
