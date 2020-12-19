from .agent import Agent
import numpy as np
import random

from ..envs.minesweeper_guided_env import MinesweeperGuidedEnv


class LocPrefAgent(Agent):
    """
    An agent using the help of the assisted environment, which always presses
    the cell with the lowest probability of being a mine. If there are cells
    that have the same probability of being a mine, it will first prefer cells
    that are on the corner, then cells that are on the edge of the board, and
    then cells that are in the interior of the board. This prioritization is
    most prevalent when starting a new game, and the first cell has to be
    opened.
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def act(self, observation, *args) -> int:
        assert np.min(observation[1]) >= 0
        if observation.ndim == 3:
            observation = observation[1]
        self.shape = observation.shape
        legal_actions = self.env.legal_actions()
        lowest_probability_actions = self.get_lowest_actions(observation,
                                                             legal_actions)

        assert len(lowest_probability_actions) >= 1

        x, y = self.action_to_coord(lowest_probability_actions)
        for coord in zip(x, y):
            if self.coord_is_corner(coord):
                return self.coord_to_action(coord)
        for coord in zip(x, y):
            if self.coord_is_edge(coord):
                return self.coord_to_action(coord)
        return random.choice(lowest_probability_actions)

    def action_to_coord(self, action):
        return np.unravel_index(action, self.shape)

    def coord_to_action(self, coord) -> int:
        return np.ravel_multi_index(coord, self.shape)

    def get_lowest_actions(self, observation, legal_actions) -> np.ndarray:
        legal_probabilities = np.ravel(observation.T)[legal_actions]
        lowest_legal_actions = np.where(
            legal_probabilities == legal_probabilities.min())
        return legal_actions[lowest_legal_actions]

    def coord_is_corner(self, coord) -> bool:
        return coord[0] in (0, self.shape[0]) and coord[1] in (0, self.shape[1])

    def coord_is_edge(self, coord) -> bool:
        return coord[0] in (0, self.shape[0]) or coord[1] in (0, self.shape[1])


# strategy like:
#   [['corner']]           -- prefer corners
#   [['corner', 'edge']]   -- prefer corners/edges
#   [['corner'], ['edge']] -- prefer corners, then edges
def locpref_strategy(strategy, game, safest):
    def cell_type(cell):
        edgex = (cell[0] in (0, game.width - 1))
        edgey = (cell[1] in (0, game.height - 1))
        if edgex and edgey:
            return 'corner'
        elif edgex or edgey:
            return 'edge'
        else:
            return 'interior'

    filtered_safest = None
    for mode in strategy:
        filtered_safest = filter(lambda e: cell_type(e) in mode, safest)
        if filtered_safest:
            break
    return filtered_safest or safest
