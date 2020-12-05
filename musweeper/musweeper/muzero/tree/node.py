import numpy as np
import random

class min_max_node_tracker:
	def __init__(self):
		self.max = 0
		self.min = 0

	def normalized(self, node_Q):
		"""
		Normalize the value to [0, 1]

		Parameters
		----------
		node_Q : float
			the node score form any node

		Returns
		-------
		float
			normalized score to [0, 1]
		"""
		return (node_Q - self.min)/ (self.max - self.min)

	def update(self, node_q):
		"""
		Update the min-max tracker

		Parameters
		----------
		node_q : float
			the node value
		"""
		self.max = max(self.max, node_q)
		self.min = min(self.min, node_q)

	def __str__(self):
		return "min : {},  max : {}".format(self.min, self.max)

	def __repr__(self):
		return self.__str__()


class node:
	def __init__(self, parrent, node_id=None, hidden_state=None):
		self.children = {}
		self.node_id = node_id
		self.parrent = parrent
		
		self.min_max_node_tracker = min_max_node_tracker() if parrent is None else parrent.min_max_node_tracker

		self._value = 0
		self.explored_count = 0
		self.wins_count = 0

		self.reward = 0
		self.policy = None
		self.prior = 0
		self.value_of_model = 0
		self.cumulative_discounted_reward = 0
		self.has_init = False

		self.hidden_state = hidden_state
		self.environment_state = None

		self.depth = 0 if parrent is None else (parrent.depth + 1)

		self.available_children_paths = None
		self.score_metric = self.upper_confidence_boundary #search_value_exploration_exploration

	def add_exploration_noise(self):
		"""
		Add exploration noise as described in the paper in Appendix C 
		"""
		dirichlet_alpha = 0.03
		root_exploration_fraction = 0.25
		actions = list(self.children.values())
		noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
		for action, noise in zip(actions, noise):
			action.prior = action.prior * (1 - root_exploration_fraction) + noise * root_exploration_fraction

	def search_value_exploration_exploration(self):
		"""
		Nodes seelection algorithm
		As described in section "Exploration and exploitation" from https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

		Returns
		-------
		float
			the ndoe score
		"""
		parrent_explored = np.log2(self.parrent.explored_count)/self.explored_count if self.parrent.explored_count != 1 and self.explored_count != 0 else 0
		child_explored = self.wins_count / self.explored_count if self.explored_count > 0 else 0
		c = np.sqrt(2)

		return child_explored + c * np.sqrt(parrent_explored)

	def upper_confidence_boundary(self):
		"""
		The upper confidene boundary
		as described in the appendix B of the paper.

		Returns
		-------
		float
			the upper confidence boundary
		"""
		self.c1 = 1.25
		self.c2 = 19652

		self.q_s_a = self.q
		self.p_s_a = self.prior

		if self.parrent is None:
			return 0

		all_actions_sum = np.sum([
			i.explored_count for i in self.parrent.children.values()
		])
		second_part = np.sqrt(
			all_actions_sum
		) / (1 + self.explored_count) * (
			self.c1  + np.log(
				 (all_actions_sum + self.c2 + 1)/ self.c2				
			)
		)
		return self.q_s_a + self.p_s_a * second_part

	@property
	def q(self):
		"""
		Calculated the node value
		As described in appendix B

		Returns
		-------
		float
			node value score
		"""
		explored = self.parrent.explored_count if self.parrent else 0
		q = self.parrent.q if self.parrent else 0
		parrent_visit_dot_parrent_q = explored * q + self.cumulative_discounted_reward
		return parrent_visit_dot_parrent_q/(explored + 1)

	@property
	def N(self):
		"""
		Calculate the node visit count 

		Returns
		-------
		int
			node visit count
		"""
		return self.parrent.explored_count + 1 if self.parrent else 0
		
	@property
	def value(self):
		return self._value

	@value.setter
	def value(self, value):
		self._value = value
		self.min_max_node_tracker.update(value)

	def on_node_creation(self, hidden_state, reward, policy, value):
		"""
		When a node is created this callback will be used

		Parameters
		----------
		hidden_state : torch.tensor
			the hidden state from the model
		reward : float
			the reward from the environment
		"""
		self.reward = reward
		self.hidden_state = hidden_state
		self.policy = policy
		self.value_of_model = value
		self.has_init = True
		
	def get_a_children_node(self, children_count):
		"""
		Returns a unexplored child node

		Parameters
		----------
		children_count : int
			the count of available children

		Returns
		-------
		node
			the new child node
		"""
		if self.available_children_paths is None:
			self.available_children_paths = list(range(children_count))
		picked_node = self.available_children_paths[random.randint(0, len(self.available_children_paths) - 1)]
		self.available_children_paths.remove(picked_node)
		return self.create_node(picked_node)

	def create_node(self, node_id):
		"""
		Create a specific child node

		Parameters
		----------
		node_id : int
			the action / node-id 

		Returns
		-------
		node
			the new node
		"""
		self.children[node_id] = node(self, node_id=node_id)
		return self.children[node_id]

	def get_children_with_id(self, node_id):
		"""
		Get node if it is a existing child node else none

		Parameters
		----------
		node_id : int
			the node id

		Returns
		-------
		node
			the newly created node
		"""
		return self.children.get(node_id, None)

	def create_children_if_not_exist(self, node_id):
		"""
		Create node if it does not exist as child

		Parameters
		----------
		node_id : int
			the node id

		Returns
		-------
		node
			the newly created node
		"""
		node = self.get_children_with_id(node_id)
		if node is None:
			return self.create_node(node_id)
		return node

	def get_best_action(self):
		"""
		Get the best available action based on children node score

		Returns
		-------
		int
			action
		"""
		return max(self.children.items(), key=lambda x: x[1].search_value_exploration_exploration())[1].node_id

	def __str__(self):
		return "id : {}, value: {}, depth: {}".format(self.node_id, self.value, self.depth)

	def __repr__(self):
		return self.__str__()
