'''
https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
'''

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
