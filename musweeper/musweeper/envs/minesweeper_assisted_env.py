from ..envs.minesweeper_guided_env import MinesweeperGuidedEnv
from gym.envs.registration import register
import numpy as np


class MinesweeperAssistedEnv(MinesweeperGuidedEnv):
    """
    An assisted minesweeper environment is a minesweeper which uses the
    probability matrix generated from the guided environment to open the cells
    that are 100% likely to not contain mines. This means that the assistant
    might do multiple steps and continue to open cells, and will only stop if
    there are no more cells that are completely safe, and it has to take a
    chance to continue the game.
    """

    def __init__(self, enable_assistance=True, **kwargs):
        super().__init__(**kwargs)
        self.enable_assistance = enable_assistance
        self.EPSILON = 1e-6

    def step(self, action):
        unnecessary_steps_before = self.unnecessary_steps
        game_over_before = self._game_over()

        (board, probability_matrix), *output = super().step(action)

        finished = (False or
                    not self.enable_assistance or
                    not self.unnecessary_steps == unnecessary_steps_before)

        while not finished:
            finished = True
            for (x, y), cell_state in np.ndenumerate(board):
                if cell_state == -1 and probability_matrix[x, y] < self.EPSILON:
                    self._open_cell(x, y)
                    finished = False
            (board, probability_matrix) = self._get_guided_observation()

        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info(action)

        self.assert_assisted_invariants(unnecessary_steps_before)

        return np.array([board, probability_matrix]), reward, done, info

    def assert_assisted_invariants(self, prev_unnecessary_steps):
        assert prev_unnecessary_steps - self.unnecessary_steps <= 1

        (board, probability_matrix) = self._get_guided_observation()
        for (x, y), cell_state in np.ndenumerate(board):
            if cell_state == -1 and probability_matrix[x, y] < self.EPSILON:
                assert False
