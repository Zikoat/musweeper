from .mock import *
from ..tree.monte_carlo import *
from ..tree.naive_search import *
from .components import *

import operator
from functools import reduce
import torch
from ..utils.debugger import *

def prod(factors):
	return reduce(operator.mul, factors, 1)

def create_model(env, testing=False, config={}):
	"""
	Creates the model based on the env layout

	Parameters
	----------
	env : gym.env
		a gym environment
	testing : bool, optional
		should a mock model be used, by default False
	config : dict, optional
		config fields, by default {}

	Returns
	-------
	tuple
		muzero model components
	"""
	representation_size = config.get("representation_size", 4)
	action_size = env.action_size if hasattr(env, 'action_size') else env.action_space.n

	if testing:
		dynamics = mock_model(custom_generator=lambda state, action: (torch.rand(representation_size, requires_grad=True), torch.rand((1), requires_grad=True)) )
		prediction = mock_model(out_features=action_size,custom_generator=lambda hidden_state: (torch.rand(action_size, requires_grad=True), torch.rand(1, requires_grad=True)) )
		representation = mock_model(custom_generator=lambda state: torch.rand(representation_size, requires_grad=True))
		return representation, dynamics, prediction
	else:
		env_size = env.state_size if not hasattr(env, 'observation_space') or env.observation_space is None else env.observation_space.shape
		env_size = env_size if type(env_size) in [int, float] else prod(env_size)

		dynamics = component_dynamics(representation_size, action_size, config)
		prediction = component_predictions(representation_size, action_size, config)
		representation = component_representation(env_size, representation_size, config)
		return representation, dynamics, prediction

class muzero(nn.Module):
	"""
	Muzero uses three components for creating it's world model, the following
	components are descried here
	- representation
	- dynamics -> returns next state and reward
	- prediction
	"""
	def __init__(self, env, representation, dynamics, prediction, max_search_depth, config={}):
		super(muzero, self).__init__()
		self.action_size = prediction.policy.out_features if type(prediction) != mock_model else prediction.out_features

		self.representation = representation
		self.dynamics = dynamics
		self.prediction = prediction

		self.tree = None
		self.max_search_depth = max_search_depth
		self.use_naive_search = config.get("use_naive_search", False)
		self.add_exploration_noise = config.get("add_exploration_noise", True) 
		self.config = config

		self.debugger = model_debugger().reset(filename_suffix="-".join([
			key + "_" + str(value) for key, value in config.items()
		]))

	def init_tree(self, state):
		"""
		Construct a tree based on state from enviorment

		Parameters
		----------
		state : torch.tensor
			the environment state

		"""
		if self.tree is None:
			internal_muzero_state = self.representation(state)
			self.tree = monte_carlo_search_tree(internal_muzero_state, self.max_search_depth, action_size=self.action_size, add_exploration_noise=self.add_exploration_noise)
			self.tree.root.environment_state = state
			self.tree.select_best_node(self.tree.root, self)

	def plan_action(self, current_state, legal_actions=None):
		"""
		at each state the model will do a roll out with the monte carlo tree and the learned model
		1 - We choose a candidate action (we will loop over all possbile actions)
		2 - Then we use monte carlo tree search based on this new state
		3 - When we find a leaf node / episode is over, we store the path in a replay buffer (used for training)
		"""
		self.init_tree(current_state)
		self.tree.set_legal_actions(legal_actions)
		self.tree.expand_node(self.tree.root, self)
		return self.tree.root

	def plan_action_naive(self, current_state):
		"""
		Use naive search for doing search

		Parameters
		----------
		current_state : torch.tensor
			current environment state

		Returns
		-------
		int	
			best action
		"""
		return naive_search(self.representation(current_state)).search(self)

	def update(self, state, action):
		"""
		Update the root node

		Parameters
		----------
		state : torch.tensor
			the next state
		action : int
			the action taken
		"""
		self.tree.update_root(state, action)

	def reset(self):
		"""
		Resets the tree
		"""
		self.tree = None

	def think(self, state):
		"""
		Use the model components to get best action

		Parameters
		----------
		state : torch.tensor
			the environment state

		Returns
		-------
		torch.tensor
			the best action
		torch.tensor
			the policy output
		"""
		policy, _ = self.prediction(self.representation(state))
		return torch.argmax(policy), policy

	def get_rollout_path(self, state, actions):
		"""
		Do a rollout with the model components

		Parameters
		----------
		state : torch.tensor
			the environment states
		actions : list
			list of actions for the rollout

		Returns
		-------
		game_event_history
			the history from the rollout
		"""
		game_history = game_event_history()
		model_world = self.representation(state)
		for a in actions:
			model_world, reward = self.dynamics(model_world, a)
			policy, value = self.prediction(model_world)
			game_history.add(
				action=torch.argmax(policy).item(),
				policy=policy,
				reward=reward,
				value=value,
				state=None
			)
		return game_history

	def save(self, optimizer, file_name=""):
		"""
		Save the model

		Parameters
		----------
		optimizer : torch.optim
			the optimizer used for learning
		"""
		torch.save({
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			"main_hidden_layer_size": self.dynamics.main_hidden_layer_size,
			'config': self.config
		}, 'muzero_{}'.format(file_name))

	def load(self, path, optimizer, cpu=False):
		"""
		load a model

		Parameters
		----------
		path : str
			path to checkpoint
		optimizer : torch.optim
			optimizer
		cpu : bool, optional
			load as cpu model, by default False

		Returns
		-------
		muzero
			returns self
		"""
		if cpu:
			checkpoint = torch.load(path, map_location=torch.device('cpu'))
		else:
			checkpoint = torch.load(path)
		self.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		return self

	def act(self, state, env):
		"""
		Make model act in a given state

		Parameters
		----------
		state : torch.tensor
			current environment state
		env : gym.env
			the environment

		Returns
		-------
		int
			the best action
		"""
		self.eval()
		self.reset()
		legal_actions = getattr(env, "legal_actions", None)
		legal_actions = legal_actions() if legal_actions is not None else None
		assert legal_actions is not None
		best_action, _ = temperature_softmax(self.plan_action(state, legal_actions), T=1, size=self.action_size, with_softmax=True)
		assert best_action in legal_actions
		return best_action
