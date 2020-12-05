import time
import random
from .node import *
import torch
from ..utils.game_history import *
import time
from graphviz import Digraph

class monte_carlo_search_tree:
	def __init__(self, root_state, max_search_depth, action_size=2, random_rollout_metric=None):
		self.max_search_depth = max_search_depth
		self.timeout = 10

		self.root = node(parrent=None, hidden_state=root_state)
		self.originale_root = self.root
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
		current_node.add_exploration_noise()

		while relative_depth_level < self.max_search_depth and not found_leaf:
			should_pick_random_node = self.random_rollout_metric(self, current_node)
			current_node_has_no_children = len(current_node.children.keys()) == 0

			if (should_pick_random_node or current_node_has_no_children) and len(current_node.children) != self.children_count:
				# TODO : add something to invalidate invalid states to make search easier when bootstraping
				new_node = current_node.get_a_children_node(self.children_count)
				if model is not None and not new_node.has_init:
					found_leaf = True
					# when a new node is found, we assign reward and policy from the model (see Expansion in Appendix B for details)
					state, action_tensor = current_node.hidden_state, torch.tensor([new_node.node_id]).float()
					next_state, reward = model.dynamics(state, action_tensor)
					policy, value_function = model.prediction(state)
					new_node.on_node_creation(next_state, reward, policy, value_function)
			else:
				new_node = self.select(current_node.children)

			current_node = new_node
			relative_depth_level += 1
		return current_node

	def backpropgate(self, leaf_node, depth, discount=0.1):
		"""
		When a leaf node is found, the values will be backpropgated and updated upwards
		"""
		cumulative_discounted_reward = 0
		visited_nodes = 0
		while leaf_node is not None:
			node_level_diff = depth - leaf_node.depth
			cumulative_discounted_reward += (discount ** (node_level_diff) * leaf_node.reward) + leaf_node.value_of_model * discount ** visited_nodes

			leaf_node.explored_count += 1
			visited_nodes += 1

			leaf_node.cumulative_discounted_reward = cumulative_discounted_reward
			leaf_node = leaf_node.parrent

		return cumulative_discounted_reward

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
			state, action_tensor = self.root.hidden_state.reshape((1, -1)), torch.tensor([node.node_id]).float().reshape((1, -1))
			next_state, reward = model.dynamics(state, action_tensor)
			policy, value_function = model.prediction(state)
			node.on_node_creation(next_state, reward, policy, value_function)

		delta_depth = 0
		while delta_depth < self.max_search_depth:
			output_node = self.expand(node, model)
			delta_depth = (output_node.depth - node.depth)
			self.backpropgate(output_node, delta_depth)
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
		assert type(action) == int, "action should be int"
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
		game_state = game_event_history() if game_state is None else game_state
		while 0 < len(current_node.children.keys()):
			best_node = max(current_node.children.items(), key=lambda x: x[1].value)[1]
			game_state.add(
				reward=best_node.reward,
				action=best_node.policy,
				value=best_node.value
			)
			current_node = best_node
		return game_state

	def construct_tree(self, dot, node, show_only_used_edges):
		if node is None:
			return None

		def create_node(node, create=True):
			depth, action = node.depth, node.node_id
			if action is None:
				action = "root"
			state = node.environment_state if node.environment_state is not None else None
			score = node.score_metric()
			parrent_action = node.parrent.node_id if node.parrent is not None else "seed"

			node_id = '{depth}_{parrent_action}_{action}_{state}'.format(depth=depth, parrent_action=parrent_action, action=action, state=state)
			if create:
				ucb = node.upper_confidence_boundary()
				ucb_reason = node.ucb_score_parts
				dot.node(node_id, 'depth {depth}, action {action}, score {score}, env state: {state}, ucb = {ucb}, ucb_reason = {ucb_reason}'.format(depth=depth, action=action, score=score, state=state, ucb=ucb, ucb_reason=ucb_reason), color='black' if state is None else 'green')
			return node_id

		for child_nodes in node.children.values():
			path_tuple = (create_node(node, create=False), create_node(child_nodes, create=False))
				
			self.construct_tree(dot, child_nodes, show_only_used_edges)
			if show_only_used_edges and "none" in path_tuple[1].lower() and "none" in path_tuple[0].lower():
				continue
			generated_current_root = create_node(node, create=True)
			dot.edge(generated_current_root, create_node(child_nodes))

	def draw(self, show_only_used_edges=False):
		dot = Digraph(comment='The search tree', format='png')
		self.construct_tree(dot, self.originale_root, show_only_used_edges)
		file_name = 'search-tree_{time}'.format(time=time.time())
		dot.render(file_name)
		return file_name + '.png'
