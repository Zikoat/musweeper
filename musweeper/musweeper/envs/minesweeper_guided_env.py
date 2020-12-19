import ast
import subprocess

import gym
from gym_minesweeper.envs import MinesweeperEnv
import numpy as np
from .mrgris.mrgris_python3 import api_solve

class SolverException(Exception):
    pass


class MinesweeperGuidedEnv(MinesweeperEnv):
    """
    A guided minesweeper environment is the same as a minesweeper
    environment, but the observation space contains an additional matrix
    which contains the probability that each cell is a mine.
    """

    def __init__(self, enable_guide=True, **kwargs):
        super().__init__(debug=False, **kwargs)
        self.enable_guide = enable_guide
        self.observation_space = gym.spaces.Box(
            low=np.float32(-2),
            high=np.float32(8),
            shape=(2, self.width, self.height))

    def step(self, action):
        observation, reward, done, info = super(MinesweeperGuidedEnv, self).step(action)
        observation = self._get_guided_observation()
        return observation, reward, done, info

    def reset(self):
        super(MinesweeperGuidedEnv, self).reset()
        observation = self._get_guided_observation()
        return observation

    def _get_guided_observation(self):
        return np.array([(super()._get_observation()),
                         self.get_probability_matrix()])

    def get_probability_matrix(self):
        if not self.enable_guide or self._game_over():
            return np.empty((self.width, self.height))

        ansi_board = self.render("ansi")
        result = api_solve({
            "board": ansi_board,
            "total_mines": self.mines_count
        })

        if "_other" in result["solution"]:
            matrix = np.full((self.width, self.height),
                             result["solution"]["_other"])
        else:
            matrix = np.empty((self.width, self.height))

        for coordinate_string, mine_probability in result["solution"].items():
            if coordinate_string != "_other":
                x, y = self._parse_solver_coordinate(coordinate_string)
                matrix[x - 1, y - 1] = mine_probability

        assert None not in matrix
        return matrix

    @staticmethod
    def _parse_solver_coordinate(coordinate):
        y, x = list(map(int, coordinate.split("-")))
        return x, y

    def legal_actions(self):
        closed_cells = super(MinesweeperGuidedEnv, self).legal_actions()
        # All the cells that are closed
        probabilities = np.ravel(self.get_probability_matrix().T)[closed_cells]
        # And are not 100% sure to be mines
        return closed_cells[probabilities != 1]