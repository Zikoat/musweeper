
class mock_model:
	def __init__(self, outputs):
		self.outputs = outputs

	def __call__(self, x):
		return outputs.pop(0)
