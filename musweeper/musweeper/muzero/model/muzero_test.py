import unittest
from .muzero import *
from .mock import *
from ..utils.basic_env import *
import torch
import numpy as np

class TestMuzero(unittest.TestCase):
	def test_plan_action(self):
		"""
		Testing that the model construct the monte carlo search tree
		"""
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)

		tree_paths = model.plan_action(env.state).children.values()
		assert len(tree_paths) == 2
		for paths in tree_paths:
			assert paths.depth == 1
			assert torch.is_tensor(paths.reward)
			assert torch.is_tensor(paths.hidden_state)
		assert model.tree.root.max_depth > 0

	def test_plan_action_leagal_actions(self):
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)

		tree_paths = list(model.plan_action(env.state, [ 0 ]).children.values())
		assert len(tree_paths) == 1
		assert tree_paths[0].node_id == 0
		assert len(tree_paths[0].children) == 2
		assert model.tree.root.max_depth > 0

		model.reset()
		tree_paths = list(model.plan_action(env.state, [ 1 ]).children.values())
		assert len(tree_paths) == 1
		assert tree_paths[0].node_id == 1
		assert len(tree_paths[0].children) == 2
		assert len(tree_paths[0].children[0].children) == 2
		assert model.tree.root.max_depth > 0

	def test_should_construct_tree_in_correct_order(self):
		env = BasicEnv()
		representation, dynamics, prediction = create_model(env, testing=True)
		max_search_depth = 3
		model = muzero(env, representation, dynamics, prediction, max_search_depth=max_search_depth)

		done = False
		depth_length = 0
		actions_history = [

		]
		while not done:
			output = model.plan_action(env.state).children.values()
			best_node = max(output, key=lambda node: node.score_metric())
			assert best_node.depth == len(actions_history) + 1
			assert sum(list(set([node.max_depth for node in output]))) > 0
			assert best_node.upper_confidence_boundary() < sum([i.upper_confidence_boundary() for i in output])
			assert np.isclose(torch.sum(model.tree.get_policy()), 1)
			assert 0 < best_node.explored_count 
			assert 0 < len(best_node.children)
			best_action = best_node.node_id
			observation, reward, done = env.step(best_action)
			model.update(observation, best_action)
			depth_length += 1
			actions_history.append(best_action)

		assert depth_length == model.tree.root.depth
		assert 0 < len(actions_history)
		assert len(list(set(actions_history))) >= 1

		current_root = model.tree.originale_root
		max_children_count = 0
		while 0 < len(actions_history):
			action = actions_history.pop(0)
			other_children = {
				key : value for key, value in current_root.children.items() if key != action
			}
			node_action = current_root.children[action]
			for _, value in other_children.items():
				assert value.score_metric() < node_action.score_metric(), "{} vs {}, level {}".format(value.score_metric(), node_action.score_metric(), node_action.depth)

			max_children_count = max(max_children_count, len(current_root.children))
				
			assert node_action.environment_state is not None, "{depth}, {env}".format(depth=node_action.depth, env=node_action.environment_state)
			assert type(node_action.score_metric()) in [int, float, np.float64]
			assert not np.isnan(node_action.score_metric())
			assert node_action.prior > 0
	
			active_nodes = list(filter(lambda x: x.environment_state is not None, node_action.children.values()))
			if len(actions_history) > 0:
				assert 1 == len(active_nodes), "bad state at {}".format(node_action.depth)
			current_root = node_action
		assert 2 == max_children_count


if __name__ == '__main__':
	unittest.main()

