from ..utils.game_history import game_event_history
import torch
import numpy as np
from ..model.components import transform_input
from ..tree.temperture import *

def play_game(model, env, self_play=False, custom_end_function=None, custom_reward_function=None, custom_state_function=None, extra_loss_tracker=None, timeout_steps=100):
	"""
	PLays a game with a given model

	Parameters
	----------
	model : Muzero
		the muzero model
	env : gym.env
		the environment
	self_play : bool, optional
		should self_play be used (not fully implemented), by default False
	custom_end_function : callable, optional
		should a custom function be used for determing if a game is done, by default None
	custom_reward_function : callable, optional
		should a custom function be used for the reward function, by default None
	custom_state_function : callable, optional
		should a custom function be used to get the observation of the enviorment, by default None
	extra_loss_tracker : callable, optional
		should a function calculate a extra loss, by default None
	timeout_steps : int, optional
		max steps to take in a environment, by default 100

	Returns
	-------
	game_event_history
		the events taken place in the game
	"""
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
			observation = custom_state_function(env) if custom_state_function is not None else observation
			state = transform_input(observation)
			info = extra_loss_tracker(env) if extra_loss_tracker else None

			if model.use_naive_search:
				best_action = model.plan_action_naive(state)
			else:
				legal_actions = getattr(env, "legal_actions", None)
				legal_actions = legal_actions() if legal_actions is not None else None
				if legal_actions is not None and len(legal_actions) == 0:
					break
				best_action = temperature_softmax(model.plan_action(state, legal_actions), T=(temperature), size=model.action_size)
				temperature *= 0.9
			observation, reward, done = env.step(best_action)[:3]

			if custom_end_function is not None:
				done = custom_end_function(env)
			if custom_reward_function is not None:
				reward = custom_reward_function(env, done)

			game_history.add(
				reward=torch.tensor([reward]).reshape((1, -1)),
				action=best_action,
				policy=None if model.use_naive_search else model.tree.get_policy(),
				value=torch.tensor([1]).reshape((1, -1)) if not done else torch.tensor([0]).reshape((1, -1)),
				state=state,
				info=info
			)
			model.reset()
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
