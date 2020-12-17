import gym
import gym_minesweeper
import time
import torch.optim as optim
from musweeper.muzero.utils.clock import clock
from musweeper.muzero.utils.training_loop import train
from musweeper.muzero.model.components import transform_input
from musweeper.muzero.model.muzero import *
from musweeper.muzero.model.selfplay import play_game, selfplay_single_player
from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.utils.training_loop import *
import numpy as np
import os
from musweeper.envs.minesweeper_guided_env import *

def get_model(env):
	representation, dynamics, prediction = create_model(env)
	model = muzero(env, representation, dynamics,
				   prediction, max_search_depth=2)
	load_model_path = "../ignore/muzero"
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
	if os.path.isfile(load_model_path):
		model.load(load_model_path, optimizer, cpu=True)
	return model, optimizer

if __name__ == "__main__":
	env = gym.make("MinesweeperGuided-v0", width=10, height=10, mine_count=10)
	
	extra_loss_tracker = lambda env: env.get_probability_matrix()
	custom_state_function = lambda env: env._get_observation()

	model, optimizer = get_model(env)

	timer = clock(60 * 30)# * 60 * 3)# * 60 * 3)
	output = train(model, env, optimizer, timer_function=lambda: timer(), custom_end_function=lambda env: env.unnecessary_steps > 0 or np.count_nonzero(np.logical_and(env.open_cells, env.mines)) > 0, extra_loss_tracker=extra_loss_tracker, custom_state_function=custom_state_function)
	model.save(optimizer)
	print(output)
	sum_lifo = 5
	print([output[i:i+sum_lifo] for i in range(0, len(output), sum_lifo)])
