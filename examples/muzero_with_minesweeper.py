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

def get_model(env, config={}):
	lr = config.get("lr", 0.001)
	weight_decay = config.get("weight_decay", 0.01)

	representation, dynamics, prediction = create_model(env, config=config)
	model = muzero(env, representation, dynamics,
				   prediction, max_search_depth=2)
	load_model_path = "../ignore/muzero"
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	if os.path.isfile(load_model_path) and config.get("load_trained", True):
		model.load(load_model_path, optimizer, cpu=True)
	return model, optimizer

if __name__ == "__main__":
	base_config = {

	}
	base_config["lr"] = 0.001
	base_config["weight_decay"] = 0

	env = gym.make("MinesweeperGuided-v0", width=10, height=10, mine_count=10)
	extra_loss_tracker = lambda env: env.get_probability_matrix()
	custom_state_function = lambda env: env._get_observation()
	model, optimizer = get_model(env, base_config)

	timer = clock(60 * 60 * 4)
	output = train(model, env, optimizer, timer_function=lambda: timer(), custom_end_function=lambda env: env.unnecessary_steps > 0 or np.count_nonzero(np.logical_and(env.open_cells, env.mines)) > 0, extra_loss_tracker=extra_loss_tracker, custom_state_function=custom_state_function)
	model.save(optimizer)
	model.debugger.save_tensorboard()

