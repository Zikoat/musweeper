from .node import *    
from .temperture import *

class naive_search:
	def __init__(self, internal_muzero_state):
		self.state = internal_muzero_state

	def get_policy(self, model):
		"""
		Get policy from the model

		Parameters
		----------
		model : Muzero
			The model

		Returns
		-------
		torch.tensor
			policy softmax
		"""
		policy, _ = model.prediction(self.state)
		policy = policy.cpu() if policy.is_cuda else policy
		output_softmax = softmax(policy.detach().numpy())
		if 1 < output_softmax.ndim:
			output_softmax = output_softmax[0]
		return output_softmax

	def search(self, model):
		"""
		Get the search policy

		Parameters
		----------
		model : Muzero
			the model

		Returns
		-------
		torch.tensor
			the softmax policy
		"""
		output_softmax = self.get_policy(model)
		return get_action(output_softmax)

