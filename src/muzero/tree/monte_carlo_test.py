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

		final_expanded_node = tree.expand()
		assert final_expanded_node.depth == max_search_depth

		current_node = root
		depth_counter = 0
		while current_node is not None:
			assert (current_node.children == None or len(current_node.children) == 1 )
			current_node = None if current_node.children is None else current_node.children[0]
			depth_counter += 1
		assert depth_counter == depth_counter

if __name__ == '__main__':
	unittest.main()

