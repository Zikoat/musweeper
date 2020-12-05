from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.model.muzero import *
from musweeper.muzero.utils.training_loop import *
from musweeper.muzero.model.components import transform_input
import time
import torch.optim as optim
import atexit
import line_profiler
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


env = BasicEnv()

representation, dynamics, prediction = create_model(env)
model = muzero(env, representation, dynamics, prediction, max_search_depth=3)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)

#@profile
def train():
	game_score = []
	sum_loss = 0

	game_replay_buffer = replay_buffer()
	#for i in range(1000):
	start = time.time()
	i = 0
	print_size = 15
	while (time.time() - start) < 60 * 60 and i < 1000:
		if i % print_size == 0 and i > 0:
			print('%d: min=%.2f median=%.2f max=%.2f eval=%.2f, sum of %d last games=%.2f, loss=%.2f' % (i, min(game_score), game_score[len(game_score)//2], max(game_score), sum(game_score)/len(game_score), print_size, sum(game_score[-print_size:]), sum_loss))
			sum_loss = 0
		model.reset()
		state = env.reset()
		done = False
		game_history = game_event_history()
		sum_score = 0
		while not done:
			output = model.plan_action(state)
			best_node = max(output, key=lambda x: x.value)
			best_action = best_node.node_id
			best_value = best_node.value

			model.tree.get_rollout_path()
			new_state, reward, done = env.step(best_action)
			
			model.update(state, best_action)
			game_history.add(
				reward=torch.tensor([reward]).reshape((1, -1)),
				action=best_action,
				value=best_value,
				state=state.reshape((1, -1))
			)
			state = new_state
			sum_score += reward

		game_replay_buffer.add(game_history)

		optimizer.zero_grad()
		total_loss = transform_input(torch.tensor(0, dtype=torch.float64))
		for game in game_replay_buffer.get_batch():
			total_loss += loss_from_game(model, game)
		total_loss.backward()
		optimizer.step()
		game_score.append(sum_score)
		sum_loss += total_loss.item()
		i += 1
	print(sum_loss)
	print(game_score)
	print(len(game_score))

train()
