import torch.nn as nn
import torch
from ..model.components import transform_input

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
		if not (t + K) < game_history.length:
			break

		"""
		TODO: I believe you shouldn't need to to a full rollout again after an epoch
		I think you can cache the original results. Try to find that out.
		"""
		total_loss = transform_input(torch.tensor(0, dtype=torch.float64))

		assert game_history.history[t].state is not None, "state should not be none {}".format(game_history.history[t])
		model.reset()
		model.plan_action(transform_input(game_history.history[t].state))
		predicted_rollout_game_history = model.tree.get_rollout_path()

		# since we moved over to UCB, it won't evenly explore
		assert predicted_rollout_game_history.length > 0, "zero output, something is wrong in the search tree"
		rollout_length = min(K, predicted_rollout_game_history.length)
		assert rollout_length > 0, "zero output, something is wrong in the search tree"
		assert model.tree.root.depth == 0, "the model shuld always start from a fresh tree"
		for k in range(rollout_length):
			predicted_reward = transform_input(predicted_rollout_game_history.history[k].reward.float())
			actual_reward = transform_input(game_history.history[t + k].reward.float())

			predicted_action = transform_input(predicted_rollout_game_history.history[k].action)
			actual_action = transform_input(torch.tensor([game_history.history[t + k].action]))

			predicted_value =  transform_input(torch.tensor([float(predicted_rollout_game_history.history[k].value)], dtype=torch.float64))
			actual_value = transform_input(torch.tensor([float(game_history.history[t + k].value)], dtype=torch.float64))

			assert 0 <= predicted_reward.item() and predicted_reward.item() <= 1, "reward should be in interval [0, 1]"
			assert 0 <= actual_reward.item() and actual_reward.item() <= 1, "reward should be in interval [0, 1]"

			assert 0 <= torch.min(predicted_action).item() and torch.max(predicted_action).item() <= model.action_size, "action should be in interval [0, action_size]"
			assert 0 <= torch.min(actual_action).item() and torch.max(actual_action).item() <= model.action_size, "action should be in interval [0, action_size]"

			assert 0 <= predicted_value.item() and predicted_value.item() <= model.action_size, "value should be in interval [0, 1]"
			assert 0 <= actual_value.item() and actual_value.item() <= model.action_size, "value should be in interval [0, 1]"

			model.prediction.debugger.log({
				"predicted_reward": str(predicted_reward),
				"reward": str(actual_reward),
			})

			model.prediction.debugger.log({
				"predicted_action": str(predicted_action),
				"action": str(actual_action),
			})

			model.prediction.debugger.log({
				"predicted_value": str(predicted_value),
				"value": str(actual_value),
			})

			total_loss += loss_error(predicted_reward, actual_reward) + loss_error_actions(predicted_action, actual_action) + loss_error(predicted_value, actual_value)
		entire_loss += scale_gradient(total_loss, 1 / rollout_length)
#		model.update(None, game_history.history[t].action)
	return entire_loss
