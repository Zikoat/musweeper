import unittest
from .muzero import *
from .mock import *
from ..utils.basic_env import *
import torch

class TestMuzero(unittest.TestCase):
	def test_plan_action(self):
		"""
		Testing that the model construct the monte carlo search tree
		"""
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)

		tree_paths = model.plan_action(env.state)
		assert len(tree_paths) == 2
		for paths in tree_paths:
			assert paths.depth - 1 == max_search_depth
			assert torch.is_tensor(paths.reward)
			assert torch.is_tensor(paths.hidden_state)

if __name__ == '__main__':
	unittest.main()

