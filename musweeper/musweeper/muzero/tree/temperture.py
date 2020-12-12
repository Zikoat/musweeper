import numpy as np

def temperature_softmax(root, T=10):
	output_softmax = create_distribution(root, T)
	return get_action(output_softmax)

def create_distribution(root, T):
	# we add + 1e-8 so that T could be set to 0 
	children_count = np.asarray([
		node.explored_count for node in root.children.values()
	]) ** 1 / (T + 1e-8)
	output_softmax = softmax(children_count)
	return output_softmax

def get_action(policy):
	actions = np.arange(len(policy))
	return np.random.choice(actions, p=policy).item()

def softmax(temperature_actions):
	e_x = np.exp(temperature_actions - np.max(temperature_actions))
	return e_x / e_x.sum()
