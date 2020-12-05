import ast
import subprocess

import gym
from gym_minesweeper.envs import MinesweeperEnv
import numpy as np


class SolverException(Exception):
    pass


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
                                                shape=(2, self.width, self.height))
        self.enable_guide = enable_guide

    def step(self, action):
        observation, *output = super(MinesweeperGuidedEnv, self).step(action)
        observation = np.array([observation, self.get_probability_matrix()])
        return observation, *output

    def get_probability_matrix(self):
        if not self.enable_guide:
            return np.empty((self.width, self.height))

        ascii_board = self.get_ascii_board()
        result = self.api_solve({"board": ascii_board, "total_mines": self.mines_count})

        if "_other" in result:
            matrix = np.array((self.width, self.height), result["_other"])
        else:
            matrix = np.empty((self.width, self.height))

        for coordinate_string, mine_probability in result["solution"].items():
            if coordinate_string != "_other":
                x, y = self.parse_coordinate(coordinate_string)
                matrix[y - 1, x - 1] = mine_probability

        assert not None in matrix
        return matrix

    def get_ascii_board(self):
        return self.render("terminal")

    def api_solve(self, payload):
        try:
            return ast.literal_eval(subprocess.run(
                [
                    "C:/users/sscho/anaconda3/envs/mrgris/python.exe",
                    "-c",
                    "from minesweepr.minesweeper_util import api_solve;"+
                    "print(api_solve({}))".format(payload)
                ],
                capture_output=True,
                check=True,
                timeout=2
            ).stdout.decode("utf-8"))
        except subprocess.CalledProcessError as e:
            raise SolverException("api_solve errored with message below:\n\n{}"
                                  .format(e.stderr.decode("utf-8")))

    def parse_coordinate(self, coordinate):
        y, x = list(map(int, coordinate.split("-")))
        return x, y