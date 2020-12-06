from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.model.muzero import *
from musweeper.muzero.utils.training_loop import *
from musweeper.muzero.model.components import transform_input
from musweeper.muzero.model.selfplay import play_game, selfplay_single_player
import time
import torch.optim as optim

class clock:
	def __init__(self, timeout):
		self.start = time.time()
		self.end = self.start + timeout

	def __call__(self):
		return self.end < time.time()

def see_model_search_tree(model, env):
	model.reset()
	done = False
	observation = env.reset()
	original_root = None
	while not done:
		output = model.plan_action(observation)
		if original_root is None:
			original_root = model.tree.root
		best_node = max(output, key=lambda node: node.value)
		best_action = best_node.node_id
		observation, reward, done = env.step(best_action)
		model.update(observation, best_action)
	return [model.tree.draw(), model.tree.draw(show_only_used_edges=True)]

def train(model, env):
	from hydra.reports.model_evaluation_report import model_report

	game_score = []
	optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)

	game_replay_buffer = replay_buffer()
	timeout = clock(30 * 1)
	i = 0
	print_interval = 15
	update_interval = 1
	selfplay_interval = 5

	report = model_report("bragearn@stud.ntnu.no")
	while not timeout() and i < 1000:
		if i % print_interval == 0 and i > 0:
			report.add_note('%d: min=%.2f median=%.2f max=%.2f eval=%.2f, sum of %d last games=%.2f' % (i, min(game_score), game_score[len(game_score)//2], max(game_score), sum(game_score)/len(game_score), print_interval, sum(game_score[-print_interval:])))
			report.add_variable("avg score last {print_interval} games".format(print_interval=print_interval), sum(game_score[-print_interval:])/print_interval)

		last_game = play_game(model, env)
		game_replay_buffer.add(last_game)
		if game_replay_buffer.is_full() and i % update_interval == 0:
			optimizer.zero_grad()
			total_loss = transform_input(torch.tensor(0, dtype=torch.float64))
			for game in game_replay_buffer.get_batch():
				total_loss += loss_from_game(model, game)
			total_loss.backward()
			optimizer.step()
			report.add_variable("loss over time",  total_loss.item())
		"""
		if i % selfplay_interval:
			optimizer.zero_grad()
			total_loss = selfplay_single_player(model, env)
			total_loss.backward()
			optimizer.step()
			report.add_variable("loss over time (selfplay)",  total_loss.item())
		"""
		sum_game_reward =  sum([
			event.reward for event in last_game.history
		])
		report.add_variable("reward over time", sum_game_reward)
		game_score.append(sum_game_reward)
		i += 1
	sample_game = play_game(model, env)
	for event in sample_game.history:
		report.add_note('state : {state}, action: {action}'.format(state=event.state, action=event.action))
	for tree in see_model_search_tree(model, env):
		report.add_plot(tree)
	for key, value in model.prediction.debugger.get_last_round():
		report.add_note('debug from predictions {}'.format(key))
		for layer in value:
			report.add_note(str(layer))
	report.add_note('first predictions')
	for el in model.prediction.debugger.logs[:12]:
		report.add_note(str(el))
	report.add_note('last predictions')
	for el in model.prediction.debugger.logs[-12:]:
		report.add_note(str(el))
	print(model.prediction.debugger.get_last_round())
	report.send_report()

if __name__ == "__main__":
	env = BasicEnv()

	representation, dynamics, prediction = create_model(env)
	model = muzero(env, representation, dynamics, prediction, max_search_depth=3)
	train(model, env)
