import gym
from gym_minesweeper.envs import MinesweeperEnv
import numpy as np


class MinesweeperGuidedEnv(MinesweeperEnv):
    """
    A guided minesweeper environment is the same as a minesweeper
    environment, but the observation space contains an additional matrix
    which contains the probability that each cell is a mine.
    """

    def __init__(self, width=8, height=8, mine_count=10, flood_fill=True, enable_guide=True):
        super().__init__(width, height, mine_count, flood_fill)
        self.observation_space = gym.spaces.Box(low=np.float32(-2),
                                                high=np.float32(8),
                                                shape=(
                                                    2, self.width, self.height))

    def step(self, action):
        observation, *output = super(MinesweeperGuidedEnv, self).step(action)
        observation = np.array([observation, self.get_probability_matrix()])
        return observation, *output

    def get_probability_matrix(self):

        return np.zeros((self.width, self.height))

