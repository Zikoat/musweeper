import unittest
from .node import *

class TestNOde(unittest.TestCase):
	def test_explored_without_win_rate_vs_unexplored(self):
		"""
		Search value should favour unexplored nodes vs already explored failed paths
		"""
		parent_node = node(None)
		parent_node.explored_count = 2

		children_node_not_explored = node(parent_node)
		
		children_node_has_explored = node(parent_node)
		children_node_has_explored.wins_count = 0
		children_node_has_explored.explored_count = 2	

		score_node_has_explored = children_node_has_explored.search_value_exploration_explotation()
		score_node_not_explored = children_node_not_explored.search_value_exploration_explotation()

		# monte carlo will faveour model that hasn't been explroed over failed models 
		assert score_node_has_explored < score_node_not_explored

	def test_explored_without_win_rate_vs_unexplored(self):
		"""
		Search value should favour unexplored nodes vs already explored failed paths
		"""
		parent_node = node(None)
		parent_node.explored_count = 2

		children_node_not_explored = node(parent_node)

		children_node_has_explored = node(parent_node)
		children_node_has_explored.wins_count = 2
		children_node_has_explored.explored_count = 2	

		score_node_has_explored = children_node_has_explored.search_value_exploration_explotation()
		score_node_not_explored = children_node_not_explored.search_value_exploration_explotation()

		# monte carlo will faveour model that has been explroed with win rate over failed models 
		assert score_node_has_explored > score_node_not_explored


if __name__ == '__main__':
	unittest.main()

