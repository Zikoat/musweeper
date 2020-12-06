from muzero_with_basic_env import see_model_search_tree
from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.model.muzero import *
from musweeper.muzero.utils.training_loop import *
from musweeper.muzero.model.components import transform_input
from musweeper.muzero.model.selfplay import play_game, selfplay_single_player
import time
import torch.optim as optim

def see_model_search_tree_quick(model, env, view_only=False):
	model.reset()
	done = False
	observation = env.reset()
	original_root = None
	actions_taken = []
	while not done:
		state = observation
		state = state if not isinstance(state, np.ndarray) else torch.from_numpy(state)
		state = torch.flatten(state) if state.dim() != 1 else state
		state = state.float()

		output = model.plan_action(observation)
		if original_root is None:
			original_root = model.tree.root
		best_node = max(output, key=lambda node: node.value)
		best_action = best_node.node_id
		observation, reward, done = env.step(best_action)[:3]
		model.update(observation, best_action)
		actions_taken.append([best_action, output])
#	return [model.tree.draw(view_only=view_only), model.tree.draw(show_only_used_edges=True, view_only=view_only)]
	return model.tree.root, actions_taken

env = BasicEnv(state_size=3)

representation, dynamics, prediction = create_model(env)
model = muzero(env, representation, dynamics, prediction, max_search_depth=3)

root, actions_taken = see_model_search_tree_quick(model, env, view_only=True)
print("best action ", list(map(lambda x: [x[0], x[1].value], root.children.items())))
print("best action id", sorted(list(map(lambda x: [x[0], x[1].value], root.children.items())), key=lambda x: x[1])[-1])
for key, val in (actions_taken):
	print(key)
	print(val)
	print("")


