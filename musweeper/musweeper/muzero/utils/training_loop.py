import torch.nn as nn

def loss_from_game(model, game_history):
	"""
	The loss tries to make the model reason about the future based on the past
	"""
	loss_error = nn.MSELoss()
	model.reset()

	entire_loss = 0
	K = model.max_search_depth
	for t in range(game_history.length):
		if not (t + K) < game_history.length:
			break

		"""
		TODO: I believe you shouldn't need to to a full rollout again after an epoch
		I think you can cache the original results. Try to find that out.
		"""
		total_loss = 0
		model.plan_action(game_history.state[t])
		predicted_rollout_game_history = model.tree.get_rollout_path()
		for k in range((K)):
			predicted_reward = predicted_rollout_game_history.reward[k]
			actual_reward = game_history.actual_reward[t + k]
			total_loss += loss_error(actual_reward, predicted_reward)
		entire_loss += (total_loss)	
		model.update(None, game_history.action[t])
	return entire_loss