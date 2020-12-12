from .mock import *
from ..tree.monte_carlo import *
from ..tree.naive_search import *
from .components import *

import operator
from functools import reduce

def prod(factors):
    return reduce(operator.mul, factors, 1)

def create_model(env, testing=False, config={}):
	representation_size = config.get("representation_size", 4)
	action_size = env.action_size if hasattr(env, 'action_size') else env.action_space.n

	if testing:
		dynamics = mock_model(custom_generator=lambda state, action: (torch.rand(representation_size, requires_grad=True), torch.rand((1), requires_grad=True)) )
		prediction = mock_model(custom_generator=lambda hidden_state: (torch.rand(action_size, requires_grad=True), torch.rand(1, requires_grad=True)) )
		representation = mock_model(custom_generator=lambda state: torch.rand(representation_size, requires_grad=True))
		return representation, dynamics, prediction
	else:
#		env_size = env.state_size if not hasattr(env, 'observation_space') else env.observation_space.shape
		env_size = env.state_size if not hasattr(env, 'observation_space') or env.observation_space is None else env.observation_space.shape
		env_size = env_size if type(env_size) in [int, float] else prod(env_size)

		dynamics = component_dynamics(representation_size)
		prediction = component_predictions(representation_size, action_size)
		representation = component_representation(env_size, representation_size)
		return representation, dynamics, prediction

class muzero(nn.Module):
	"""
	Muzero uses three components for creating it's world model, the following
	components are descried here
	- representation
	- dynamics -> returns next state and reward
	- prediction
	"""
	def __init__(self, env, representation, dynamics, prediction, max_search_depth, add_exploration_noise=True, use_naive_search=False):
		super(muzero, self).__init__()
		self.action_size = prediction.policy.out_features if type(prediction) != mock_model else 2

		self.representation = representation
		self.dynamics = dynamics
		self.prediction = prediction

		self.max_search_depth = max_search_depth
		self.tree = None
		self.use_naive_search = use_naive_search
		self.add_exploration_noise = add_exploration_noise

	def init_tree(self, state):
		if self.tree is None:
			internal_muzero_state = self.representation(state)
			self.tree = monte_carlo_search_tree(internal_muzero_state, self.max_search_depth, action_size=self.action_size, add_exploration_noise=self.add_exploration_noise)
			self.tree.root.environment_state = state
			self.tree.select_best_node(self.tree.root, self)
		else:
			return None
		return self.tree

	def plan_action(self, current_state):
		"""
		at each state the model will do a roll out with the monte carlo tree and the learned model
		1 - We choose a candidate action (we will loop over all possbile actions)
		2 - Then we use monte carlo tree search based on this new state
		3 - When we find a leaf node / episode is over, we store the path in a replay buffer (used for training)
		"""
		self.init_tree(current_state)
		self.tree.expand_node(self.tree.root, self)
		return self.tree.root

	def plan_action_naive(self, current_state):
		return naive_search(self.representation(current_state)).search(self)

	def update(self, state, action):
		self.tree.update_root(state, action)

	def reset(self):
		self.tree = None

	def think(self, state):
		policy, _ = self.prediction(self.representation(state))
		return torch.argmax(policy), policy

	def get_rollout_path(self, state, actions):
		game_history = game_event_history()
		model_world = self.representation(state)
		for a in actions:
			model_world, reward = self.dynamics(model_world, a)
			#action, policy = model.think(observation)
			#best_action = action.item()
			#observation, reward, done = env.step(best_action)[:3]
			policy, value = self.prediction(model_world)
			game_history.add(
				action=policy,
				reward=reward,
				value=value,
				state=None
			)
		return game_history

	def save(self, optimizer):
		torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'muzero')

	def load(self, path, optimizer, cpu=False):
		if cpu:
			checkpoint = torch.load(path, map_location=torch.device('cpu'))
		else:
			checkpoint = torch.load(path)
		self.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		return self
