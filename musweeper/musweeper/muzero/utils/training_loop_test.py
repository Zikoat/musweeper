import unittest
from .training_loop import *
import unittest
from ..model.muzero import *
from ..model.mock import *
from .basic_env import *
from ..model.selfplay import play_game
import torch
import numpy as np

class TestTrainingLoop(unittest.TestCase):
	def test_model_debugger(self):
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)
		output = play_game(model, env)

		loss = loss_from_game(model, output)
		assert torch.is_tensor(loss)
		assert loss.dim() == 0
		assert loss != 0

if __name__ == '__main__':
	unittest.main()
