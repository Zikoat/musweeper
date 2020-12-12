import unittest
from .monte_carlo import *
from .node import *
from ..utils.basic_env import *
from ..model.mock import *
from ..model.muzero import *

class TestMonteCarlo(unittest.TestCase):
	def setUp(self):
		env = BasicEnv()
		max_search_depth = 3
		# even if rollout is False, it will do rollout since none of the nodes has a children (yet)
		self.tree = monte_carlo_search_tree(None, max_search_depth=max_search_depth, random_rollout_metric=lambda tree, node: False)
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		self.model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)

	def test_should_expand_tree(self):
		"""
		Search value should favour unexplored nodes vs already explored failed paths
		"""
		root = node(None)
		root.min_max_node_tracker.update(0)
		root.min_max_node_tracker.update(10)
		root.hidden_state = torch.rand((4, 4))

		assert root.min_max_node_tracker.min == 0
		assert root.min_max_node_tracker.max == 10

		max_search_depth = 3
		# even if rollout is False, it will do rollout since none of the nodes has a children (yet)
		tree = monte_carlo_search_tree(None, max_search_depth=max_search_depth)
		for _ in range(max_search_depth ** 2):
			tree.expand(root, self.model)

	def test_should_update_correctly(self):
		"""
		Search value should favour unexplored nodes vs already explored failed paths
		"""
		root = node(None)
		root.min_max_node_tracker.update(0)
		root.min_max_node_tracker.update(10)

		assert root.min_max_node_tracker.min == 0
		assert root.min_max_node_tracker.max == 10

		max_search_depth = 3
		# even if rollout is False, it will do rollout since none of the nodes has a children (yet)
		tree = monte_carlo_search_tree(
			None, max_search_depth=max_search_depth, random_rollout_metric=lambda tree, node: False)
		tree.root = root

		tree.expand(root, self.model)
		assert len(root.children.values()) > 0
		assert 0 == tree.root.upper_confidence_boundary()
		assert root == tree.root

		tree.update_root(None, tree.root.get_best_action())

		assert (root.depth + 1) == tree.root.depth
		# since we break on first leaf node, there should be no children on new root
		assert len(tree.root.children) == 0
		assert root.max_depth > 0

	def test_monte_carlo_backpropgate(self):
		"""
		The nodes should be updated after an node backpropgates.
		"""
		root = node(None)
		root.min_max_node_tracker.update(0)
		root.min_max_node_tracker.update(10)
		root.hidden_state = torch.rand((4, 4))

		assert root.min_max_node_tracker.min == 0
		assert root.min_max_node_tracker.max == 10

		final_expanded_node = self.tree.expand(root, self.model)

		"""
		setting all reward to one to make it easy to verify
		"""
		current_node = final_expanded_node
		while current_node is not None:
			current_node.reward = 1
			current_node.value_of_model = 1
			current_node = current_node.parrent

		assert 0 < self.tree.backpropgate(
			final_expanded_node, start_depth=0, discount=1)
		for child_nodes in root.children.values():
			assert 0 < child_nodes.upper_confidence_boundary()
		assert root.max_depth > 0

	# skipped for now as the functionality is changed
	def that_node_values_propgate_correctly_with_model(self):
		max_search_depth = 1
		env = BasicEnv()
		for best_action in range(2):
			env.reset()
			# output new state (as internal state representation) and reward
			dynamics = mock_model(name="dynamics", outputs=[
				[torch.tensor([0, 1]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0])],
				[torch.tensor([0, 1]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0])],
				[torch.tensor([0, 1]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0])],
				#		[torch.tensor([0, 1]), torch.tensor([1])],
				#		[torch.tensor([0, 0]), torch.tensor([0])],
			])
			# output policy, value
			prediction = mock_model(name="prediction", outputs=[
				[torch.tensor([0, 0]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0.2])],
				[torch.tensor([0, 0]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0.2])],
				[torch.tensor([0, 0]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0.2])],
				[torch.tensor([0, 0]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0.2])],
			])
			for i in prediction.outputs:
				i[0][best_action] = 1
			# output internal state representation
			representation = mock_model(name="representation", outputs=[
				torch.tensor([0, 1])  # created before the search starts

			])

			model = muzero(env, representation, dynamics,
						   prediction, max_search_depth, add_exploration_noise=False)
			output = model.plan_action(env.reset())
			assert len(output) == 2
			assert max(output, key=lambda x: x.score_metric()).node_id == best_action
			for component in [prediction, dynamics, representation]:
				assert len(component.outputs) == 0, component
	
	def test_should_prune_search_tree(self):
		dynamics = mock_model(name="dynamics", outputs=[
			[torch.tensor([0, 1]), torch.tensor([1])],
			[torch.tensor([0, 0]), torch.tensor([0])],
			[torch.tensor([0, 1]), torch.tensor([1])],
			[torch.tensor([0, 0]), torch.tensor([0])],
			[torch.tensor([0, 1]), torch.tensor([1])],
			[torch.tensor([0, 0]), torch.tensor([0])],
			#		[torch.tensor([0, 1]), torch.tensor([1])],
			#		[torch.tensor([0, 0]), torch.tensor([0])],
		])
		# output policy, value
		prediction = mock_model(name="prediction", outputs=[
			[torch.tensor([0, 0]), torch.tensor([1])],
			[torch.tensor([0, 0]), torch.tensor([0.2])],
			[torch.tensor([0, 0]), torch.tensor([1])],
			[torch.tensor([0, 0]), torch.tensor([0.2])],
			[torch.tensor([0, 0]), torch.tensor([1])],
			[torch.tensor([0, 0]), torch.tensor([0.2])],
			[torch.tensor([0, 0]), torch.tensor([1])],
			[torch.tensor([0, 0]), torch.tensor([0.2])],
		])

		# output internal state representation
		representation = mock_model(name="representation", outputs=[])

		max_search_depth = 3
		model = muzero(None, representation, dynamics,
					   prediction, max_search_depth=max_search_depth)

		root = node(None)
		root.min_max_node_tracker.update(0)
		root.min_max_node_tracker.update(10)

		child = node(root)
		child.node_id = 1
		root.children[child.node_id] = child

		top_k_nodes_to_search = 1
		tree = monte_carlo_search_tree(None, max_search_depth=max_search_depth,
									   top_k_nodes_to_search=top_k_nodes_to_search, random_rollout_metric=lambda tree, node: False)
		tree.root = root
		tree.set_values_for_expand_a_node(child, model)

#		assert len(root.children) == top_k_nodes_to_search
		assert np.isclose(torch.sum(tree.get_policy()), 1)

if __name__ == '__main__':
	unittest.main()
