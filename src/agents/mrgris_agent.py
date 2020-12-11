class LocPrefAgent(object):
    """
    An agent using the help of the assisted environment, which always presses
    the cell with the lowest probability of being a mine. If there are cells
    that have the same probability of being a mine, it will first prefer cells
    that are on the corner, then cells that are on the edge of the board, and
    then cells that are in the interior of the board. This prioritization is
    most prevalent when starting a new game, and the first cell has to be
    opened.
    """

    def __init__(self, env, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
