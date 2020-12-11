from unittest import TestCase
import gym
import numpy as np
from src.envs.minesweeper_assisted_env import MinesweeperAssistedEnv

class TestMinesweeperAssistedEnv(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("MinesweeperAssisted-v0")

    def setUp(self) -> None:
        self.env.reset()

    def test_no_sure_cells_left(self):
        ob, reward, episode_over, info = self.env.step(self.env.action_space.sample())
        if not episode_over:
            normal_observation = ob[0]
            guide_observation = ob[1]
            # todo find any cells that have 0 probability and are closed
            assert not np.allclose(1, guide_observation)

    def test_first_open_is_mine(self):
        env = gym.make("MinesweeperAssisted-v0", seed=0)
        env.step(31)
        self.assertIn("B", env.render("ansi"))

    def test_opens_multiple(self):
        env = gym.make("MinesweeperAssisted-v0", seed=2)
        ob, reward, episode_over, info = env.step(0)
        print(info["unnecessary steps"])
        self.assertFalse(episode_over)
        self.assertEqual(48, info["opened cells"])
        self.assertEqual(0.8888888888888888, reward)
        self.assertEqual(0, info["unnecessary steps"])

        ob, reward, episode_over, info = env.step(0)
        self.assertFalse(episode_over)
        self.assertEqual(48, info["opened cells"])
        self.assertLess(reward, 0.8888888888888888)
        self.assertEqual(1, info["unnecessary steps"])
