import torch.nn as nn
import torch
from ..model.components import transform_input

def loss_from_game(model, game_history):
	"""
	The loss tries to make the model reason about the future based on the past
	"""
	loss_error = nn.MSELoss()
	loss_error_actions = nn.CrossEntropyLoss()
	model.reset()

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
		model.plan_action(transform_input(game_history.state[t]))
		predicted_rollout_game_history = model.tree.get_rollout_path()
		assert len(predicted_rollout_game_history.reward) >= K, "wrong size {} expected {}".format(len(predicted_rollout_game_history.reward), K)
		for k in range((K)):
			predicted_reward = transform_input(predicted_rollout_game_history.reward[k].float())
			actual_reward = transform_input(game_history.actual_reward[t + k].float())

			predicted_action = transform_input(predicted_rollout_game_history.action[k])
			actual_action = transform_input(torch.tensor([game_history.actual_action[t + k]]))

			predicted_value =  transform_input(torch.tensor([float(predicted_rollout_game_history.value[k])], dtype=torch.float64))
			actual_value = transform_input(torch.tensor([float(game_history.actual_value[t + k])], dtype=torch.float64))

			total_loss += loss_error(predicted_reward, actual_reward) + loss_error_actions(predicted_action, actual_action) + loss_error(predicted_value, actual_value)
		entire_loss += total_loss	
		model.update(None, game_history.actual_action[t])
	return entire_loss
