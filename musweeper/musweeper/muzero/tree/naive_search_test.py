import unittest

from .node import *
from .temperture import *
from .naive_search import *
from ..model.components import *
from ..model.muzero import *
from ..utils.basic_env import *

class TestNaiveSearch(unittest.TestCase):
	def test_search(self):
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)
		model.plan_action_naive(env.reset())
		
if __name__ == "__main__":
	TestNaiveSearch()
