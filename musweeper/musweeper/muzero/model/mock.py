import torch

class mock_model:
	"""
	Basic mock model
	Used instead of having to create a torch module.
	"""
	def __init__(self, outputs=None, output_dim=None, custom_generator=None):
		if outputs is not None:
			self.outputs = outputs
		elif output_dim is not None:
			self.output_dim = output_dim
		elif custom_generator is not None:
			self.custom_generator = custom_generator
		else:
			raise Exception("need to specify outputs or output_dim")

	def __call__(self, *args):
		if hasattr(self, 'outputs'):
			fifo = self.outputs.pop(0)
			output = fifo if not type(fifo) == list else list(map(lambda x: x.float(), fifo))
			return output
		elif hasattr(self, 'custom_generator'):
			return self.custom_generator(*args)
		else:
			return torch.rand(self.output_dim)
