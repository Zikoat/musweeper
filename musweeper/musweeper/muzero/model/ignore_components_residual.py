import torch
import torch.nn as nn
import torch.nn.functional as F
from.components import shared_backbone, transform_input

num_filters = 8
num_blocks = 2

class ConvBlock(nn.Module):
	def __init__(self, filters, kernel_size, bn=False):
		super().__init__()
		self.conv = nn.Conv2d(filters[0], filters[1], kernel_size, stride=1, padding=kernel_size//2, bias=False)
		self.bn = nn.BatchNorm2d(filters[-1]) if bn else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		return x

class ResidualBlock(nn.Module):
	def __init__(self, filters):
		super().__init__()
		self.conv = ConvBlock((filters, filters), 3, True)

	def forward(self, x):
		return F.relu(x + (self.conv(x)))


class component_dynamics_residual(nn.Module, shared_backbone):
	"""
	Description of the dynamics component of Muzero 
	the role of the dynamics component is to learn the effect of actions
	in diffrent states.
	"""
	def __init__(self, representation_size):
		super(component_dynamics_residual, self).__init__()
		shared_backbone.__init__(self)
		self.shared_hidden_size = 16

		self.layer0 = ConvBlock((representation_size + 1, num_filters), 3, bn=True)
		self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

#		self.preprocess_state = nn.Linear(representation_size, self.shared_hidden_size).to(self.device)
#		self.preprocess_action = nn.Linear(1, self.shared_hidden_size).to(self.device)
#
#		self.combined = nn.Linear(2 * self.shared_hidden_size, 128).to(self.device)
#
#		self.reward = nn.Linear(128, 1).to(self.device)
#		self.new_state = nn.Linear(128, representation_size).to(self.device)
#
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
			action = torch.tensor([action])

		if len(action.shape) == 1:
			action = action.reshape((1, -1))

		state = transform_input(state)
		action = transform_input(action)
		h = torch.cat([state, action], dim=1)
		h = self.layer0(h)
		for block in self.blocks:
			h = block(h)
		return h
		
#		state = self.preprocess_state(state)
#		state = torch.sigmoid(state.view(state.size(0), -1))
#		action = self.preprocess_action(action)
#		action = torch.sigmoid(action.view(action.size(0), -1))
#
#		combined = torch.sigmoid(self.combined(torch.cat((state, action), dim=1)))
#		self.debugger.stop_track_time("dynamics")
#		return torch.sigmoid(self.new_state(combined)), torch.sigmoid(self.reward(combined))

class component_predictions_residual(nn.Module, shared_backbone):
	"""
	The prediction component will learn the optimal state and value function
	this is predicted from the current state from the representation component.
	"""
	def __init__(self, representation_size, action_size):
		super(component_predictions_residual, self).__init__()
		shared_backbone.__init__(self)

		main_hidden_layer_size = 16
		second_hidden_layer_size = main_hidden_layer_size // 2

#		self.preprocess_state = nn.Linear(representation_size, main_hidden_layer_size).to(self.device)
#		self.combined = nn.Linear(main_hidden_layer_size, second_hidden_layer_size).to(self.device)

		self.conv_p1 = ConvBlock((2, 4), 1, bn=True)
		self.conv_p2 = ConvBlock((4, 1), 1)

		self.conv_v = ConvBlock((num_filters, 4), 1, bn=True)
		self.fc_v = nn.Linear(representation_size * 4, 1, bias=False)

		self.action_size = action_size
		self.board_size = representation_size
#		self.value = nn.Linear(second_hidden_layer_size, 1).to(self.device)
#		self.policy = nn.Linear(second_hidden_layer_size, action_size).to(self.device)

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

		h_p = F.relu(self.conv_p1(state))
		h_p = self.conv_p2(h_p).view(-1, self.action_size)

		h_v = F.relu(self.conv_v(state))
		h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

		# range of value is -1 ~ 1
		return F.softmax(h_p, dim=-1), torch.tanh(h_v)		

#		state = torch.sigmoid(self.preprocess_state(state))
#		combined = torch.sigmoid(self.combined(state))
#		policy, value = torch.softmax(torch.sigmoid(self.policy(combined)), dim=1), torch.sigmoid(self.value(combined))
#		return policy, value

class component_representation_residual(nn.Module, shared_backbone):
	"""
	The representation component will convert the real environment state to lantent space
	muzero does not use environment state directly. It converts it to latent space and
	uses that in the other components.
	"""
	def __init__(self, env_size, representation_size):
		super(component_representation_residual, self).__init__()
		shared_backbone.__init__(self)

		self.preprocess_state = ConvBlock((env_size, num_filters), 3, bn=True)
		self.output_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])
		self.output_state = ConvBlock((num_filters, 1), 3, bn=True)

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
		state = F.relu(self.preprocess_state(state))
		for block in self.output_blocks:
			state = block(state)
		return self.output_state(state)
