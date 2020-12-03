import time
import random
from .node import *
import torch
from ..utils.game_history import *

class clock:
	def __init__(self, timeout):
		self.start = time.time()
		self.end = self.start + timeout

	def __call__(self):
		return self.start < self.end

class monte_carlo_search_tree:
	def __init__(self, root_state, max_search_depth, action_size=2, random_rollout_metric=None):
		self.max_search_depth = max_search_depth
		self.timeout = 10

		self.root = node(parrent=None, hidden_state=root_state)
		self.children_count = action_size

		# coin-flip
		self.random_rollout_metric = (lambda tree, node: random.randint(0, 2) == 1) if random_rollout_metric is None else random_rollout_metric

	def expand(self, current_node=None, model=None):
		"""
		Expand the search tree

		Parameters
		----------
		current_node : node, optional
			the seed node, by default None
		model : model, optional
			model used to update the reward from a given model, by default None

		Returns
		-------
		node
			leaf node after search
		"""
		relative_depth_level = 0
		current_node = self.root if current_node is None else current_node
		found_leaf = False
		# TODO : add "or" for if current node is a terminated state 
		while relative_depth_level < self.max_search_depth:# and clock(self.timeout):
			should_pick_random_node = self.random_rollout_metric(self, current_node)
			current_node_has_no_children = len(current_node.children.keys()) == 0

			if (should_pick_random_node or current_node_has_no_children) and len(current_node.children) != self.children_count:
				# TODO : add something to invalidate invalid states to make search easier when bootstraping
				new_node = current_node.get_a_children_node(self.children_count)
				# TODO : you should actually backpropgate as soon as a leaf is found
				found_leaf = True
			else:
				new_node = self.select(current_node.children)

			if model is not None:
				new_node.on_node_creation(*model.dynamics(current_node.hidden_state, torch.tensor([new_node.node_id]).float().reshape((1, -1))))

			current_node = new_node
			relative_depth_level += 1
		return current_node

	def backpropgate(self, leaf_node, discount=0.1):
		"""
		When a leaf node is found, the values will be backpropgated and updated upwards
		"""
		value = 0
		while leaf_node is not None:
			leaf_node.value += value
			leaf_node.explored_count += 1

			value = leaf_node.reward + discount * value
			leaf_node = leaf_node.parrent
		return value

	def select(self, nodes):
		"""
		Select best node

		Parameters
		----------
		nodes : list
			list of nodes

		Returns
		-------
		node
			best node
		"""
		return max(list(nodes.values()), key=lambda node: node.score_metric())

	def expand_node(self, node, model):
		"""
		Expand node

		Parameters
		----------
		node : node
			seed node
		model : muzero
			model used for predictions of reward

		Returns
		-------
		node
			the leaf node
		"""
		if type(node) == int:
			node = self.root.create_children_if_not_exist(node)
		if model:
			state, action = self.root.hidden_state.reshape((1, -1)), torch.tensor([node.node_id]).float().reshape((1, -1))
			node.on_node_creation(*model.dynamics(state, action))
		output_node = self.expand(node, model)
		self.backpropgate(output_node)
		return output_node

	def update_root(self, state, action):
		"""
		Update root for when a action is taken

		Parameters
		----------
		state : torch
			the new environment state
		action : int
			the action taken.
		"""
		self.root = self.root.children[action]
		self.root.environment_state = state

	def get_rollout_path(self, game_state=None):
		"""
		Get the rollout path taken for the tree

		Parameters
		----------
		game_state : game_history, optional
			used if you want to append to history, by default None

		Returns
		-------
		game_history
			the game history from the search tree
		"""
		current_node = self.root
		game_state = game_history() if game_state is None else game_state
		while 0 < len(current_node.children.keys()):
			best_node = max(current_node.children.items(), key=lambda x: x[1].value)[1]
			game_state.reward.append(best_node.reward)
			game_state.value.append(best_node.value)			
			current_node = best_node
		return game_state
