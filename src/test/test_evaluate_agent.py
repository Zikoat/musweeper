from unittest import TestCase
import gym.wrappers.tests
import gym_minesweeper
from ..agents.random_agent import RandomAgent
from src.evaluate_agent import evaluate_agent


class TestEvaluateAgent(TestCase):
    def test_evaluate_agent(self):
        env = gym.make("Minesweeper-v0")
        env.reset()
        agent = RandomAgent(env.action_space)
        evaluate_agent(agent, env)
