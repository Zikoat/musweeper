from abc import abstractmethod


class Agent(object):
    @abstractmethod
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def act(self, observation, reward, done):
        pass
