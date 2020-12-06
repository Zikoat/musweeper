import unittest
from .monte_carlo import *
from .node import *
from ..utils.basic_env import *
from ..model.mock import *
from ..model.muzero import *

class TestMonteCarlo(unittest.TestCase):
	def test_should_expand_tree(self):
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
		tree = monte_carlo_search_tree(None, max_search_depth=max_search_depth, random_rollout_metric=lambda tree, node: False)
		final_expanded_node = root
		for _ in range(max_search_depth ** 2):
			final_expanded_node = tree.expand(final_expanded_node)
#		assert final_expanded_node.depth == max_search_depth

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
		root.min_max_node_tracker.update(0)
		root.min_max_node_tracker.update(10)

		assert root.min_max_node_tracker.min == 0
		assert root.min_max_node_tracker.max == 10

		max_search_depth = 3
		# even if rollout is False, it will do rollout since none of the nodes has a children (yet)
		tree = monte_carlo_search_tree(None, max_search_depth=max_search_depth, random_rollout_metric=lambda tree, node: False)
		tree.root = root

		final_expanded_node = tree.expand(root)
		assert final_expanded_node.depth == max_search_depth
		assert len(root.children.values()) > 0
		assert 0 == tree.root.upper_confidence_boundary()
		assert root == tree.root
		tree.update_root(None, tree.root.get_best_action())

		assert (root.depth + 1) == tree.root.depth
		assert len(tree.root.children) > 0
		# since no model is connect we expect the value to be zero.	
		values = [child_nodes.upper_confidence_boundary() for child_nodes in tree.root.children.values()]
		assert 1 == len(values)
		assert 0 == values[0]
		assert 0  == tree.root.upper_confidence_boundary()

	def test_monte_carlo_backpropgate(self):
		"""
		The nodes should be updated after an node backpropgates.
		"""
		root = node(None)
		root.min_max_node_tracker.update(0)
		root.min_max_node_tracker.update(10)

		assert root.min_max_node_tracker.min == 0
		assert root.min_max_node_tracker.max == 10

		max_search_depth = 3
		# even if rollout is False, it will do rollout since none of the nodes has a children (yet)
		tree = monte_carlo_search_tree(None, max_search_depth=max_search_depth, random_rollout_metric=lambda tree, node: False)
		final_expanded_node = root
		for _ in range(max_search_depth ** 2):
			final_expanded_node = tree.expand(final_expanded_node)

		"""
		setting all reward to one to make it easy to verify
		"""
		current_node = final_expanded_node
		while current_node is not None:
			current_node.reward = 1
			current_node.value_of_model = 1
			current_node = current_node.parrent
		# since the reward = 1 and discount=1, the total value should be the sum of length
		#expected_reward = (max_search_depth + 1) * 2
		
		assert 0 < tree.backpropgate(final_expanded_node, depth=max_search_depth, discount=1)
		for child_nodes in root.children.values():
			assert 0 < child_nodes.upper_confidence_boundary()

	def test_that_node_values_propgate_correctly_with_model(self):
		max_search_depth = 1
		env = BasicEnv()
		for best_action in range(2):
			env.reset()
			# output new state (as internal state representation) and reward
			dynamics = mock_model(outputs=[
				[torch.tensor([0, 1]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0])],
				[torch.tensor([0, 1]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0])],
				[torch.tensor([0, 1]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0])],
				[torch.tensor([0, 1]), torch.tensor([1])],
				[torch.tensor([0, 0]), torch.tensor([0])],
			])
			# output policy, value 
			prediction = mock_model(outputs=[
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
			representation = mock_model(outputs=[
				torch.tensor([0, 1]) # created before the search starts

			])

			model = muzero(env, representation, dynamics, prediction, max_search_depth)
			output = model.plan_action(env.reset())
			assert len(output) == 2
			assert max(output, key=lambda x: x.score_metric()).node_id == best_action
			

if __name__ == '__main__':
	unittest.main()

