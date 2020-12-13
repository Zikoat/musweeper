
from musweeper.agents import random_agent
from musweeper.agents.random_agent import *
from musweeper.evaluate_agent import *
import gym
import gym_minesweeper
from muzero_with_minesweeper import get_model as get_muzero_model

env = gym.make("Minesweeper-v0", width=5, height=5, mine_count=5)
get_muzero_model(env)[0]

evaluate_agent(RandomAgent(env.action_space), env)
#evaluate_agent(get_muzero_model(env)[0], env)
