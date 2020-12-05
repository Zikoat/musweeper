from ..utils.game_history import game_event_history
import torch

def play_game(model, env, self_play=False):
	model.reset()
	observation = env.reset()
	game_history = game_event_history()
	done = False
	while not done:
		if self_play:
			action, policy = model.think(observation)
			best_action = action.item()
			observation, reward, done = env.step(best_action)
			game_history.add(
				action=policy,
				reward=None,
				value=None,
				state=None
			)
		else:
			state = observation
			output = model.plan_action(observation)
			best_node = max(output, key=lambda node: node.value)
			best_value = best_node.value
			best_action = best_node.node_id

			observation, reward, done = env.step(best_action)
			game_history.add(
				reward=torch.tensor([reward]).reshape((1, -1)),
				action=best_action,
				value=best_value,
				state=state.reshape((1, -1))
			)
			model.update(None, best_action)
	return game_history

def selfplay_single_player(model, env, games=10):
	"""
	Since the model will is playing a single player game
	the self play algorithm needs to be adjusted to account for this

	Parameters
	----------
	model : muzero
		muzero - since we are working with only one model (since it's single player) no need for model storage
	env : gym.Env
		the environment where the model will be playing
	"""
	history = []
	for _ in range(games):
		history.append(play_game(model, env, self_play=True))
	sorted_history = sorted(history, key=lambda x: x.historic_reward)
	middle = len(sorted_history) // 2
	bad_plays = sorted_history[:middle]
	win_plays = sorted_history[middle:]
	loss = 0
	for player, sign in zip([bad_plays, win_plays], [-1, 1]):
		for game in player:
			for event in game.history:
				loss += (sign * event.action).sum()
	return loss
	