import numpy as np

def temperature_softmax(root, T=10, size=None, with_softmax=False):
	"""
	Create a temperature softmax based on children exploration count from root node

	Parameters
	----------
	root : node
		the root node
	T : int, optional
		temperature parameter, by default 10
	size : int, optional
		action size, by default None
	with_softmax : bool, optional
		if softmax should be returned, by default False

	Returns
	-------
	tuple or int
		action with softmax or only best action
	"""
	output_softmax, valid = create_distribution(root, T, size)
	action = get_action(output_softmax)

	if with_softmax:
		return action, output_softmax
	return action 

def create_distribution(root, T, size=None):
	"""
	Create a distribution based on the children exploration node

	Parameters
	----------
	root : node
		the root node to select children nodes from
	T : int
		temperature parameter
	size : int, optional
		softmax size, by default None

	Returns
	-------
	tuple
		softmax and child node exploration array
	"""
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
	"""
	Get action based on policy

	Parameters
	----------
	policy : numpy.array
		the distribution to sample action from

	Returns
	-------
	int
		action sampled from distribution
	"""
	policy /= policy.sum()
	actions = np.arange(len(policy))
	return np.random.choice(actions, p=policy).item()

def softmax(temperature_actions):
	"""
	Create a softmax

	Parameters
	----------
	temperature_actions : nd.array
		the array to create a softmax from

	Returns
	-------
	nd.array
		softmax output
	"""
	zero_index = temperature_actions == 0
	temperature_actions[zero_index] = 0
	e_x = np.exp(temperature_actions - np.max(temperature_actions))
	output = np.zeros((temperature_actions.shape))
	output[~zero_index] = e_x[~zero_index] / e_x[~zero_index].sum()
	return output
