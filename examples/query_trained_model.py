from muzero_with_minesweeper import get_model as get_muzero_model
import numpy as np
import gym
import gym_minesweeper
import time
from musweeper.muzero.tree.temperture import *
from musweeper.muzero.tree.naive_search import *


def pad_size(arr, size):
	return arr + [0, ] * (size - len(arr))

# https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
def bmatrix(a):
	"""Returns a LaTeX bmatrix

	:a: numpy array
	:returns: LaTeX bmatrix as a string
	"""
	a = np.round(a, 4)
	if len(a.shape) > 2:
		raise ValueError('bmatrix can at most display two dimensions')
	lines = str(a).replace('[', '').replace(']', '').splitlines()
	rv = [r'\begin{bmatrix}']
	rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
	rv +=  [r'\end{bmatrix}']
	return '\n'.join(rv)

def how_does_muzero_view_blank_board(model, env):
	state = env.reset()
	distro, searched = create_distribution(model.plan_action(state), T=1)
	print(bmatrix(distro.reshape(state.shape)))
	print(distro)

	state = env.reset()
	distro = naive_search(model.representation(state)).get_policy(model)
	print(bmatrix(distro.reshape(state.shape)))
	print(distro)

def how_does_muzero_view_common_strategy(model, env):
	state_0 = np.asarray([
		[-1, ] * 10,
		pad_size([-1, -1, 1, 1], 10),
		pad_size([1, 1, 1], 10),
	] + list(pad_size([], 10) for i in range(7)))
	assert state_0.shape == (10, 10)
	env.reset()
	env.open_cells = state_0

	distro, searched = create_distribution(model.plan_action(env.open_cells, legal_actions=env.legal_actions()), T=1, size=100)
	print(distro.shape)
	print(bmatrix(distro.reshape(state_0.shape)[:5, :5]))

if __name__ == "__main__":
	env = gym.make("Minesweeper-v0", width=10, height=10, mine_count=10)
	model = get_muzero_model(env)[0]
#	how_does_muzero_view_blank_board(model, env)
	how_does_muzero_view_common_strategy(model, env)

