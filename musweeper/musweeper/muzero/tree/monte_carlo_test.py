import unittest
from .monte_carlo import *
from .node import *

class TestMonteCarlo(unittest.TestCase):
	def test_should_expand_tree(self):
		"""
		Search value should favour unexplored nodes vs already explored failed paths
		"""
		root = node(None)
		max_search_depth = 3
		# even if rollout is False, it will do rollout since none of the nodes has a children (yet)
		tree = monte_carlo_search_tree(root, max_search_depth=max_search_depth, random_rollout_metric=lambda tree, node: False)

		for i in range(max_search_depth ** 2):
			final_expanded_node = tree.expand()
		assert final_expanded_node.depth == max_search_depth

		current_node = root
		depth_counter = 0
		while current_node is not None:
			assert (current_node.children == {} or len(current_node.children) == 1 )
			current_node = None if len(current_node.children.keys()) == 0 else current_node.children[list(current_node.children.keys())[0]]
			depth_counter += 1
		assert depth_counter == depth_counter

	def test_should_update_correctly(self):
		"""
		Search value should favour unexplored nodes vs already explored failed paths
		"""
		root = node(None)
		max_search_depth = 3
		# even if rollout is False, it will do rollout since none of the nodes has a children (yet)
		tree = monte_carlo_search_tree(root, max_search_depth=max_search_depth, random_rollout_metric=lambda tree, node: False)

		for i in range(max_search_depth ** 2):
			final_expanded_node = tree.expand()
		assert final_expanded_node.depth == max_search_depth
		tree.update_root(None, tree.root.get_best_action())

		assert final_expanded_node.depth == max_search_depth


	def test_monte_carlo_backpropgate(self):
		"""
		The nodes should be updated after an node backpropgates.
		"""
		root = node(None)
		max_search_depth = 3
		# even if rollout is False, it will do rollout since none of the nodes has a children (yet)
		tree = monte_carlo_search_tree(root, max_search_depth=max_search_depth, random_rollout_metric=lambda tree, node: False)
		for i in range(max_search_depth ** 2):
			final_expanded_node = tree.expand()

		"""
		setting all reward to one to make it easy to verify
		"""
		current_node = final_expanded_node
		while current_node is not None:
			current_node.reward = 1
			current_node.value_of_model = 1
			current_node = current_node.parrent
		# since the reward = 1 and discount=1, the total value should be the sum of length
		expected_reward = (max_search_depth + 1) * 2
		assert expected_reward == tree.backpropgate(final_expanded_node, depth=max_search_depth, discount=1)

if __name__ == '__main__':
	unittest.main()

