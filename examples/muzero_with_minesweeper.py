import gym
import gym_minesweeper
from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.model.muzero import *
from musweeper.muzero.utils.training_loop import *
from musweeper.muzero.model.components import transform_input
from musweeper.muzero.model.selfplay import play_game, selfplay_single_player
import gym

env = gym.make("Minesweeper-v0")

representation, dynamics, prediction = create_model(env)
model = muzero(env, representation, dynamics, prediction, max_search_depth=3)

history = play_game(model, env)

#env.reset()
#output = env.step(env.action_space.sample())
#output = env.step(env.action_space.sample())

#print(env.render('ansi'))
#print(output)
