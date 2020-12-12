import time

import torch.optim as optim
from musweeper.muzero.utils.clock import clock
from musweeper.muzero.utils.training_loop import train
from musweeper.muzero.model.components import transform_input
from musweeper.muzero.model.muzero import *
from musweeper.muzero.model.selfplay import play_game, selfplay_single_player
from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.utils.training_loop import *

if __name__ == "__main__":
	env = BasicEnv(state_size=2)
	representation, dynamics, prediction = create_model(env)
	model = muzero(env, representation, dynamics,
				   prediction, max_search_depth=2)
	optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

	timer = clock(60 * 10)
	output = train(model, env, optimizer, timer_function=lambda: timer())
	print(output)
	model.save(optimizer)
