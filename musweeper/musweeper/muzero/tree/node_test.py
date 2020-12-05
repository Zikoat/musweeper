import unittest
from .node import *

class TestNode(unittest.TestCase):
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

        score_node_has_explored = children_node_has_explored.search_value_exploration_exploration()
        score_node_not_explored = children_node_not_explored.search_value_exploration_exploration()

        # monte carlo will faveour model that hasn't been explored over failed models
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

        score_node_has_explored = children_node_has_explored.search_value_exploration_exploration()
        score_node_not_explored = children_node_not_explored.search_value_exploration_exploration()

        # monte carlo will faveour model that has been explored with win rate over failed models
        assert score_node_has_explored > score_node_not_explored


    def test_explored_ucb(self):
        """
        Search value should favour explored nodes vs already explored failed paths
        """
        parent_node = node(None)
        parent_node.explored_count = 2
        parent_node.value = 10
        parent_node.cumulative_discounted_reward = 4

        childreN_node_with_high_prior = node(parent_node)
        childreN_node_with_high_prior.prior = 0.8
        childreN_node_with_high_prior.wins_count = 3
        childreN_node_with_high_prior.explored_count = 3

        children_node_has_explored = node(parent_node)
        children_node_has_explored.wins_count = 2
        children_node_has_explored.explored_count = 2
        children_node_has_explored.value = 42
        children_node_has_explored.prior = 0.01

        score_node_has_explored = children_node_has_explored.upper_confidence_boundary()
        score_node_with_high_prior = childreN_node_with_high_prior.upper_confidence_boundary()

        # monte carlo will faveour model that has been explored with win rate over failed models
        assert score_node_with_high_prior < score_node_has_explored 

    def test_explored_with_high_win_rate_vs_low(self):
        """
        Search value should favour unexplored nodes vs already explored failed paths
        """
        parent_node = node(None)
        parent_node.explored_count = 2
        parent_node.value = 10
        parent_node.cumulative_discounted_reward = 4

        children_node_high_win = node(parent_node)
        children_node_high_win.wins_count = 3
        children_node_high_win.explored_count = 5
        children_node_high_win.prior = 0.01

        children_node_low_win = node(parent_node)
        children_node_low_win.wins_count = 2
        children_node_low_win.explored_count = 51
        children_node_low_win.value = 0
        children_node_low_win.prior = 0.01

        score_node_low_win = children_node_low_win.upper_confidence_boundary()
        score_node_high_win = children_node_high_win.upper_confidence_boundary()

        # monte carlo will faveour model that hasn't been explored over failed models
        assert score_node_low_win < score_node_high_win

    def test_ucb_non_parrent_should_give_zero(self):
        parent_node = node(None)
        assert parent_node.upper_confidence_boundary() == 0        

    def test_node_min_max_tracker(self):
        parent_node = node(None)
        parent_node.explored_count = 1
        parent_node.value = 10

        child_node = node(parent_node)
        child_node.explored_count = 1
        child_node.value = 20

        assert parent_node.min_max_node_tracker.min == 10
        assert parent_node.min_max_node_tracker.max == 20

        assert parent_node.min_max_node_tracker.normalized(10) == 0
        assert parent_node.min_max_node_tracker.normalized(15) == 0.5
        assert parent_node.min_max_node_tracker.normalized(20) == 1

        # same object should be shared
        assert child_node.min_max_node_tracker.normalized(10) == 0
        assert child_node.min_max_node_tracker.normalized(20) == 1


if __name__ == '__main__':
    unittest.main()
