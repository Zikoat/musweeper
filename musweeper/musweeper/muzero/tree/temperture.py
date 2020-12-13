import numpy as np

def temperature_softmax(root, T=10, size=None):
	output_softmax, _ = create_distribution(root, T, size)
	return get_action(output_softmax)

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
	actions = np.arange(len(policy))
	return np.random.choice(actions, p=policy).item()

def softmax(temperature_actions):
	e_x = np.exp(temperature_actions - np.max(temperature_actions))
	return e_x / e_x.sum()
