import torch
import torch.nn as nn

from ..model.components import transform_input
from ..model.selfplay import play_game
from .game_history import replay_buffer


def scale_gradient(tensor, scale):
	return tensor * scale + tensor.detach() * (1 - scale)


def loss_from_game(model, game_history):
	"""
	The loss tries to make the model reason about the future based on the past
	"""
	loss_error = nn.MSELoss()
	loss_error_actions = nn.CrossEntropyLoss()

	entire_loss = transform_input(torch.tensor(0, dtype=torch.float64))
	K = model.max_search_depth
	for t in range(game_history.length):
		total_loss = transform_input(torch.tensor(0, dtype=torch.float64))
		assert game_history.history[t].state is not None, "state should not be none {}".format(
			game_history.history[t])

		model.reset()
		model.plan_action(transform_input(game_history.history[t].state))
		predicted_rollout_game_history = model.tree.get_rollout_path()
		min_max_normalize = model.tree.root.min_max_node_tracker

		assert predicted_rollout_game_history.length > 0, "zero output, something is wrong in the search tree"
		rollout_length = min(K, predicted_rollout_game_history.length)

		assert rollout_length > 0, "zero output, something is wrong in the search tree"
		assert model.tree.root.depth == 0, "the model shuld always start from a fresh tree"
		size = game_history.history[t].state.shape[-1]
		for k in range(rollout_length):
			if not (t + k) < game_history.length:
				break
			predicted_reward = transform_input(
				predicted_rollout_game_history.history[k].reward.float()) / size
			actual_reward = transform_input(
				game_history.history[t + k].reward.float())
			actual_reward = (
				actual_reward[0] if actual_reward.dim() > 1 else actual_reward) / size

			predicted_action = transform_input(
				predicted_rollout_game_history.history[k].action)
			actual_action = transform_input(
				torch.tensor([game_history.history[t + k].action]))

			predicted_value = transform_input(torch.tensor([float(min_max_normalize.normalized(
				predicted_rollout_game_history.history[k].value))], dtype=torch.float64))
			actual_value = transform_input(torch.tensor(
				[float(game_history.history[t + k].value)], dtype=torch.float64))

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

			total_loss += loss_error(predicted_reward, actual_reward) + loss_error_actions(
				predicted_action.reshape(1, -1), actual_action.long()) + loss_error(predicted_value, actual_value)
		entire_loss += scale_gradient(total_loss, 1 / rollout_length)
	return entire_loss


def train(model, env, optimizer, timer_function, log=False, print_interval=15, update_interval=15, custom_end_function=None, custom_reward_function=None):
	game_replay_buffer = replay_buffer()

	i = 0
	game_score = []
	while not timer_function():
		if 0 < i and i % print_interval == 0:
			print('%d: min=%.2f median=%.2f max=%.2f eval=%.2f, sum of %d last games=%.2f' % (i, min(game_score), game_score[len(
				game_score)//2], max(game_score), sum(game_score)/len(game_score), print_interval, sum(game_score[-print_interval:])))

		last_game = play_game(model, env, custom_end_function=custom_end_function, custom_reward_function=custom_reward_function)
		game_replay_buffer.add(last_game)
		if game_replay_buffer.is_full() and i % update_interval == 0:
			optimizer.zero_grad()
			total_loss = transform_input(torch.tensor(0, dtype=torch.float64))
			for game in game_replay_buffer.get_batch():
				total_loss += loss_from_game(model, game)
			total_loss.backward()
			optimizer.step()
		game_score.append(last_game.historic_reward)
		i += 1
	return game_score
