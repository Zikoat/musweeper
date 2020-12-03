from collections import deque
import random

class game_history:
	def __init__(self):
		self.reward = []
		self.actual_reward = []

		self.state = []

		self.action = []
		self.actual_action = []

		self.actual_value = []
		self.value = []

	@property
	def length(self):
		return len(self.state)

class reply_buffer:
	def __init__(self):
		self.memory = deque(maxlen=2000)
		self.batch_size = 32

	def add(self, game):
		self.memory.append(game)

	def get_batch(self):
		if len(self.memory) < self.batch_size:
			return self.memory
		return random.sample(self.memory, self.batch_size)
