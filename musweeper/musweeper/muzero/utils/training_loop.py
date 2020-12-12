import torch
import torch.nn as nn
import numpy as np

from ..model.components import transform_input
from ..model.selfplay import play_game
from .game_history import replay_buffer



def scale_gradient(tensor, scale):
	return tensor * scale + tensor.detach() * (1 - scale)


def loss_from_game(model, game_history, debug=False):
	"""
	The loss tries to make the model reason about the future based on the past
	"""
	loss_error = nn.MSELoss()
	loss_error_actions = nn.CrossEntropyLoss()

	entire_loss = transform_input(torch.tensor(0, dtype=torch.float64))
	K = model.max_search_depth
	length = int(game_history.length)
	full_game_history = list(game_history.history)
	for t in range(length):
		total_loss = transform_input(torch.tensor(0, dtype=torch.float64))
		assert full_game_history[t].state is not None, "state should not be none {}".format(
			full_game_history[t])

		had_to_fill = False
		if (t + K) < length:
			for _ in range((t + K), length):	
				full_game_history.append(game_history.add(
					reward = torch.tensor([0]),
					action = np.random.randint(2),
					value = 0,
					state = None,
					soft=True,
				))
			had_to_fill = True
		predicted_rollout_game_history = model.get_rollout_path(full_game_history[t].state, [
			full_game_history[tk].action for tk in range(min(t + K, game_history.length))
		])
		
		assert predicted_rollout_game_history.length > 0, "zero output, something is wrong in the search tree"

		for k in range(K):
			predicted_reward = transform_input(predicted_rollout_game_history.history[k].reward.float())
			actual_reward = transform_input(full_game_history[t + k].reward.float())
			actual_reward = (actual_reward[0] if actual_reward.dim() > 1 else actual_reward)

			predicted_action = transform_input(
				predicted_rollout_game_history.history[k].action)
			actual_action = transform_input(
				torch.tensor([full_game_history[t + k].action]))

			predicted_value = transform_input(torch.tensor([float(predicted_rollout_game_history.history[k].value)], dtype=torch.float64))
			actual_value = transform_input(torch.tensor(
				[float(full_game_history[t + k].value)], dtype=torch.float64))

			predicted_value = predicted_reward
			actual_value = actual_reward

			assert 0 <= predicted_reward.item() and predicted_reward.item(
			) <= 1, "reward should be in interval [0, 1] {}".format(predicted_reward)
			assert 0 <= actual_reward.item() and actual_reward.item(
			) <= 1, "reward should be in interval [0, 1] {}".format(actual_reward)

			assert 0 <= torch.min(predicted_action).item() and torch.max(predicted_action).item(
			) <= model.action_size, "action should be in interval [0, action_size]"
			assert 0 <= torch.min(actual_action).item() and torch.max(actual_action).item(
			) <= model.action_size, "action should be in interval [0, action_size]"

			assert 0 <= predicted_value.item() and predicted_value.item(
			) <= 1, "value should be in interval [0, 1], got {}".format(predicted_value)
			assert 0 <= actual_value.item() and actual_value.item(
			) <= 1, "value should be in interval [0, 1], got {}".format(actual_value)

			loss_reward = loss_error(predicted_reward, actual_reward)
			#loss_action = torch.sum(-actual_action * torch.log(predicted_action))
			loss_action = loss_error_actions(predicted_action.reshape(1, -1), actual_action.long())
			loss_value = loss_error(predicted_value, actual_value)
			if debug:
				print("reward : {} ({}, {}), action : {}({}, {}), value: {}".format(loss_reward, predicted_reward, actual_reward, loss_action, predicted_action, actual_action, loss_value) )

			model.prediction.debugger.write_to_tensorboard("predicted_reward", predicted_reward.item(), None)
			model.prediction.debugger.write_to_tensorboard("actual_reward", actual_reward.item(), None)

			model.prediction.debugger.write_to_tensorboard("reward_loss", loss_reward, None)
			model.prediction.debugger.write_to_tensorboard("action_loss", loss_action, None)
			model.prediction.debugger.write_to_tensorboard("action_value", loss_value, None)

			total_loss += loss_reward + loss_action + loss_value
		entire_loss += scale_gradient(total_loss, 1 / K)
	#	print(entire_loss)
		debug = False
		if had_to_fill:
			break
#		model.reset()
	return entire_loss


def train(model, env, optimizer, timer_function, log=False, print_interval=15, update_interval=15, custom_end_function=None, custom_reward_function=None):
	game_replay_buffer = replay_buffer()

	i = 0
	game_score = []
	last_loss = None
	while not timer_function():
		print_debug = 0 < i and i % print_interval == 0
		if print_debug:
			tree_depth = model.prediction.debugger.variable_log.get("max_depth", None)
			best_explored = model.prediction.debugger.variable_log.get("best_explored", None)
			print('%d: min=%.2f median=%.2f max=%.2f eval=%.2f, sum of %d last games=%.2f, loss=%s, depth=%s, explored=%s' % (i, min(game_score), game_score[len(
				game_score)//2], max(game_score), sum(game_score)/len(game_score), print_interval, sum(game_score[-print_interval:]), last_loss, tree_depth, best_explored))
			model.prediction.debugger.write_to_tensorboard("avg_score", sum(game_score[-print_interval:])/print_interval, None)

		last_game = play_game(model, env, custom_end_function=custom_end_function, custom_reward_function=custom_reward_function)
		game_replay_buffer.add(last_game)
		if game_replay_buffer.is_full() and i % update_interval == 0:
			# we loop over and get 3 batches instead of doing many small updates
			# (smaller loss, model hopefully learns better)
			for _ in range(16):
				optimizer.zero_grad()
				total_loss = transform_input(torch.tensor(0, dtype=torch.float64))
				for game in game_replay_buffer.get_batch():
					total_loss += loss_from_game(model, game, debug=print_debug)
				total_loss.backward()
				last_loss = total_loss.item()
				optimizer.step()
				model.prediction.debugger.write_to_tensorboard("loss", last_loss, None)
		model.prediction.debugger.write_to_tensorboard("game", last_game.historic_reward, i)
		game_score.append(last_game.historic_reward)
		i += 1
	return game_score
