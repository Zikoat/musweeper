import random


# https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
from .agent import Agent

class RandomAgent(Agent):
    """The world's simplest agent!"""
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = env.action_space
        self.has_legal_actions = callable(
            getattr(self.env, "legal_actions", None))

    def act(self, *args):
        if self.has_legal_actions:
            return random.choice(self.env.legal_actions())
        else:
            return self.action_space.sample()

