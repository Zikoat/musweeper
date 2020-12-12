from .node import *    
from .temperture import *

class naive_search:
	def __init__(self, internal_muzero_state):
		self.state = internal_muzero_state

	def select_best_node(self, **args):
		pass

	def search(self, model):
		policy, _ = model.prediction(self.state)
		print(policy.detach().numpy())
		print(len(policy.detach().numpy()))
		output_softmax = softmax(policy.detach().numpy())
		return get_action(output_softmax)

