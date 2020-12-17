from .node import *    
from .temperture import *

class naive_search:
	def __init__(self, internal_muzero_state):
		self.state = internal_muzero_state
		self.reset()

	def select_best_node(self, **args):
		pass

	def reset(self):
		self.actions = {
		
		}

	def get_policy(self, model):
		policy, _ = model.prediction(self.state)
		policy = policy.cpu() if policy.is_cuda else policy
		output_softmax = softmax(policy.detach().numpy())
		if 1 < output_softmax.ndim:
			output_softmax = output_softmax[0]
		return output_softmax

	def search(self, model, depth=3):
		output_softmax = self.get_policy(model)
		return get_action(output_softmax)

