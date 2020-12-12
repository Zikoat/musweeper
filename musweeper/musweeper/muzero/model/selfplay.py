from ..utils.game_history import game_event_history
import torch
import numpy as np
from ..model.components import transform_input
from ..tree.temperture import *

def play_game(model, env, self_play=False, custom_end_function=None, custom_reward_function=None, timeout_steps=100):
	model.reset()
	observation = env.reset()
	game_history = game_event_history()
	done = False
	step = 0
	temperature = 1
	while not done and step < timeout_steps:
		if self_play:
			action, policy = model.think(observation)
			best_action = action.item()
			observation, reward, done = env.step(best_action)[:3]
			game_history.add(
				action=policy,
				reward=None,
				value=None,
				state=None
			)
		else:
			model.prediction.debugger.start_track_time("game play thinking")
			state = transform_input(observation)
			best_action = temperature_softmax(model.plan_action(state), T=(temperature))
			temperature *= 0.9
#			best_node = max(output, key=lambda node: node.value)
#			best_value = model.tree.root.min_max_node_tracker.normalized(best_node.value)
#			best_action = best_node.node_id
#			model.prediction.debugger.stop_track_time("game play thinking")
			
#			model.prediction.debugger.start_track_time("game play action")
			observation, reward, done = env.step(best_action)[:3]
	#		model.prediction.debugger.stop_track_time("game play action")
			if custom_end_function is not None:
				done = custom_end_function(env)
			if custom_reward_function is not None:
				reward = custom_reward_function(env, done)
			game_history.add(
				reward=torch.tensor([reward]).reshape((1, -1)),
				action=best_action,
				value=0,
				state=state.reshape((1, -1))
			)
			model.prediction.debugger.variable_log["max_depth"] = model.tree.root.max_depth
			model.prediction.debugger.variable_log["best_explored"] = model.tree.root.children[best_action].explored_count
			model.reset()
			#model.update(None, best_action)
		step += 1
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
	