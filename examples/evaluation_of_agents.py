
from musweeper.agents import random_agent
from musweeper.agents.mrgris_agent import *
from musweeper.agents.random_agent import *
from musweeper.evaluate_agent import *
import gym
import gym_minesweeper
from muzero_with_minesweeper import get_model as get_muzero_model
import matplotlib.pyplot as plt
import os
import json
from musweeper.muzero.tree.temperture import *


output_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_path, "output/")
json_cache_file = os.path.join(output_path, "cahced_info.json")

def evaluation_vs_radom_agent(cached={}):
	env = gym.make("Minesweeper-v0", width=10, height=10, mine_count=10)

	random_stats = cached.get("random_stats", None)
	muzero_stats = cached.get("muzero_stats", None)
	if random_stats is None:
		random_stats = evaluate_agent(RandomAgent(env), env)[:, 0].tolist()
	if muzero_stats is None:
		random_stats = evaluate_agent(get_muzero_model(env)[0], env)

	random_stats_sum = np.cumsum(random_stats)
	muzero_stats_sum = np.cumsum(muzero_stats)

	_, ax = plt.subplots()
	x = np.arange(random_stats_sum.shape[0])
	ax.plot(x, random_stats_sum, '--r', label='Random agent')
	ax.plot(x, muzero_stats_sum, '-b', label='Muzero agent')
	leg = ax.legend()
	plt.title("Muzero vs random agent")
	plt.plot()

	path = os.path.join(output_path, "eval_random_agent.png")
	plt.savefig(path)

	return {
		"random_stats": random_stats,
		"muzero_stats": muzero_stats,	
	}

def get_cache(file):
	if os.path.isfile(file):
		return json.loads(open(file, "r").read())
	return {}

def get_probability_matrix_blank_board():
	env = gym.make("MinesweeperGuided-v0", width=10, height=10, mine_count=10)
	env.reset()

	observation = env._get_observation()
	prob_matrix = env.get_probability_matrix()

	muzero_prob_matrix, _ = create_distribution(get_muzero_model(env)[0].plan_action(observation, legal_actions=env.legal_actions()), T=1, size=100)

	for board, name in zip([muzero_prob_matrix, prob_matrix], ["muzero", "mrgris"]):
		plt.clf()
		plt.imshow(board.reshape((10, 10)), cmap='gray')
		plt.colorbar()
		path = os.path.join(output_path, "blank_board_{}.png".format(name))
		plt.savefig(path)

if __name__ == "__main__":
#	data = evaluation_vs_radom_agent(get_cache(json_cache_file))
#	open(json_cache_file, "w").write(json.dumps(data))
	get_probability_matrix_blank_board()
