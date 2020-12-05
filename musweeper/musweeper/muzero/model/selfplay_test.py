import unittest
from .selfplay import *
import torch
from ..utils.basic_env import *
from .muzero import *

class TestSelfplay(unittest.TestCase):
	def test_should_expand_tree(self):
		"""
		Search value should favour unexplored nodes vs already explored failed paths
		"""
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)

		loss = selfplay_single_player(model, env)
		assert torch.is_tensor(loss)
		assert loss.requires_grad, "requires grad ? {}".format(loss.requires_grad)
		assert loss.item() != 0

if __name__ == '__main__':
	unittest.main()

