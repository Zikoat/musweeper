from unittest import TestCase
import gym.wrappers.tests
import gym_minesweeper

from src.agents.mrgris_agent import LocPrefAgent
from src.agents.random_agent import RandomAgent
from src.evaluate_agent import evaluate_agent
import numpy as np
import pytest


@pytest.mark.parametrize(
    "env_name, agent_class",
    [("Minesweeper-v0", RandomAgent),
     ("MinesweeperGuided-v0", LocPrefAgent),
     ("MinesweeperAssisted-v0", LocPrefAgent)])
def test_evaluate_random_agent(env_name, agent_class):
    env = gym.make(env_name)
    env.reset()
    agent = agent_class(env)
    stats = evaluate_agent(agent, env)
    assert np.all(stats[:, 3] == 0)
