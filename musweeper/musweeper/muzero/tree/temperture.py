import numpy as np

def temperature_softmax(root, T=10, size=None, with_softmax=False, get_legal_only=False):
	output_softmax, valid = create_distribution(root, T, size)
	action = get_action(output_softmax)

	#if get_legal_only:
	#	while valid[action] == 0:
	#		action = get_action(output_softmax)

	if with_softmax:
		return action, output_softmax
	return action 

def create_distribution(root, T, size=None):
	# we add + 1e-8 so that T could be set to 0 
	if size is not None:
		children_count = np.zeros((size))
		for key, node in root.children.items():
			children_count[key] = node.explored_count
	else:
		children_count = np.asarray([
			node.explored_count for node in root.children.values()
		]) 
	children_count = children_count ** 1 / (T + 1e-8)
	output_softmax = softmax(children_count)

	return output_softmax, children_count

def get_action(policy):
	policy /= policy.sum()
	actions = np.arange(len(policy))
	return np.random.choice(actions, p=policy).item()

def softmax(temperature_actions):
	zero_index = temperature_actions == 0
	temperature_actions[zero_index] = 0
	e_x = np.exp(temperature_actions - np.max(temperature_actions))
	output = np.zeros((temperature_actions.shape))
	output[~zero_index] = e_x[~zero_index] / e_x[~zero_index].sum()
	return output
