import unittest

import numpy as np
import torch

from ..model.mock import *
from ..model.muzero import *
from ..model.selfplay import play_game
from .basic_env import *
from .training_loop import *


class TestTrainingLoop(unittest.TestCase):
	def test_model_monte_carlo(self):
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)
		output = play_game(model, env)

		loss = loss_from_game(model, output)
		assert torch.is_tensor(loss)
		assert loss.dim() == 0
		assert loss != 0

	def test_model_naive_search(self):
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth, config={
			"use_naive_search":True
		})

		output = play_game(model, env)

		loss = loss_from_game(model, output)
		assert torch.is_tensor(loss)
		assert loss.dim() == 0
		assert loss != 0

if __name__ == '__main__':
	unittest.main()
