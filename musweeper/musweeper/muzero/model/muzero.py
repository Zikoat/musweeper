from .mock import *
from ..tree.monte_carlo import *
from .components import *

def create_model(env, testing=False, config={}):
	representation_size = config.get("representation_size", 4)
	env_size = env.state_size
	action_size = env.action_size

	if testing:
		dynamics = mock_model(custom_generator=lambda state, action: (torch.rand(representation_size), torch.rand((1))) )
		prediction = mock_model(custom_generator=lambda hidden_state: (torch.rand(1), torch.rand(1)) )
		representation = mock_model(custom_generator=lambda state: torch.rand(representation_size))
		return representation, dynamics, prediction

	else:
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
	def __init__(self, env, representation, dynamics, prediction, max_search_depth):
		super(muzero, self).__init__()
		self.action_size = env.action_size

		self.representation = representation
		self.dynamics = dynamics
		self.prediction = prediction

		self.max_search_depth = max_search_depth
		self.tree = None

	def plan_action(self, current_state):
		"""
		at each state the model will do a roll out with the monte carlo tree and the learned model
		1 - We choose a candidate action (we will loop over all possbile actions)
		2 - Then we use monte carlo tree search based on this new state
		3 - When we find a leaf node / episode is over, we store the path in a replay buffer (used for training)
		"""
		internal_muzero_state = self.representation(current_state)
		if self.tree is None:
			self.tree = monte_carlo_search_tree(internal_muzero_state, self.max_search_depth, action_size=self.action_size)
			self.tree.root.environment_state = current_state
		# loop over all possible actions from current node
		return [
			self.tree.expand_node(node, self) for node in range(self.action_size)
		]

	def update(self, state, action):
		self.tree.update_root(state, action)

	def reset(self):
		self.tree = None

	def train(self, replay_buffer):
		pass

