'''
https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
'''
import random


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.has_legal_actions = callable(getattr(self.env, "legal_actions", None))

    def act(self, observation, reward, done):
        if self.has_legal_actions:
            return random.choice(self.env.legal_actions())
        else:
            return self.action_space.sample()
