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

    def __init__(self, enable_guide=True, **kwargs):
        super().__init__(debug=False, **kwargs)
        self.enable_guide = enable_guide
        self.observation_space = gym.spaces.Box(low=np.float32(-2),
                                                high=np.float32(8),
                                                shape=(2, self.width, self.height))

    def step(self, action):
        observation, *output = super(MinesweeperGuidedEnv, self).step(action)
        observation = self._get_guided_observation()
        return observation, *output

    def _get_guided_observation(self):
        return np.array([(super()._get_observation()),
                         self.get_probability_matrix()])

    def get_probability_matrix(self):
        if not self.enable_guide or self._game_over():
            return np.empty((self.width, self.height))

        ansi_board = self.get_ansi_board()
        result = self.api_solve({"board": ansi_board, "total_mines": self.mines_count})
        print(result)
        if "_other" in result["solution"]:
            matrix = np.full((self.width, self.height), result["solution"]["_other"])
        else:
            matrix = np.empty((self.width, self.height))

        for coordinate_string, mine_probability in result["solution"].items():
            if coordinate_string != "_other":
                x, y = self.parse_coordinate(coordinate_string)
                matrix[x - 1, y - 1] = mine_probability

        assert not None in matrix
        return matrix

    def get_ansi_board(self):
        return self.render("ansi")

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
            print(e.stdout)
            raise SolverException("api_solve errored with message below:\n\n{}"
                                  .format(e.stderr.decode("utf-8")))

    def parse_coordinate(self, coordinate):
        y, x = list(map(int, coordinate.split("-")))
        return x, y

    def legal_actions(self):
        closed_cells = super(MinesweeperGuidedEnv, self).legal_actions()
        # All the cells that are closed
        probabilities = np.ravel(self.get_probability_matrix().T)[closed_cells]
        # And are not 100% sure to be mines
        return closed_cells[probabilities != 1]
