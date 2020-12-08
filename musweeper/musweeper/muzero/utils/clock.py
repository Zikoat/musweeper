import time

class clock:
	def __init__(self, timeout):
		self.start = time.time()
		self.end = self.start + timeout

	def __call__(self):
		return self.end < time.time()
