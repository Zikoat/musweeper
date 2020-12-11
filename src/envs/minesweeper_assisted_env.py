from src.envs.minesweeper_guided_env import MinesweeperGuidedEnv
from gym.envs.registration import register
import numpy as np


class MinesweeperAssistedEnv(MinesweeperGuidedEnv):
    """
    An assisted minesweeper environment is a minesweeper which uses the probability matrix generated from the guided environment to open the cells that are 100% likely to not contain mines. This means that the assistant might do multiple steps and continue to open cells, and might even finish the game on its own, but it will stop if it has to take a chance.
    todo rewrite description of assisted minesweeper env.
    """

    def __init__(self, enable_assistance=True, **kwargs):
        super().__init__(**kwargs)
        self.enable_assistance = enable_assistance
        self.EPSILON = 1e-6

    def step(self, action):
        unnecessary_steps_before = self.unnecessary_steps
        game_over_before = self._game_over()

        (board, probability_matrix), *output = super().step(action)

        finished = False or not self.enable_assistance or not (self.unnecessary_steps == unnecessary_steps_before)
        print("starting", finished)
        while not finished:
            print("looping")
            finished = True
            for (x, y), cell_state in np.ndenumerate(board):
                if cell_state == -1 and probability_matrix[x, y] < self.EPSILON:
                    self._open_cell(x, y)
                    print("opening")
                    finished = False
            (board, probability_matrix) = self._get_guided_observation()

        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info(game_over_before, action)

        if unnecessary_steps_before - self.unnecessary_steps != 0:
            print("unnecessary steps changed. 1:{}, 2:{}".format(unnecessary_steps_before, self.unnecessary_steps))
        # todo move unnecessary steps increment check to assert invariants
        # todo refactor this so it says "if unnecessary steps changed, the board and probability matrix should not change, and assert that unnecessary steps did not change.
        print(probability_matrix)
        print(board)

        return np.array([board, probability_matrix]), reward, done, info

    def assert_assisted_invariants(self):
        assert not 0 in self.get_probability_matrix() # todo assert that when step is done, there are no 0% cells not opened.

register(
    id='MinesweeperAssisted-v0',
    entry_point='src.envs.minesweeper_assisted_env:MinesweeperAssistedEnv',
    nondeterministic=True,
)