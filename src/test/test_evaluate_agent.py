from unittest import TestCase
import gym.wrappers.tests
import gym_minesweeper
from src.agents.random_agent import RandomAgent
from src.evaluate_agent import evaluate_agent
import numpy as np

class TestEvaluateAgent(TestCase):
    def test_evaluate_random_agent(self):
        env = gym.make("Minesweeper-v0")
        env.reset()
        agent = RandomAgent(env)
        stats = evaluate_agent(agent, env)
        self.assertTrue(np.all(stats[:,3] == 0))
