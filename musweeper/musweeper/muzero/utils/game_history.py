from collections import deque
import random
from collections import namedtuple
import torch

class game_event_history:
	def __init__(self):
		self.history = []
		self.event = namedtuple('event', ['reward', 'action', 'value', 'state'])
		self.historic_reward = 0

	def add(self, reward, action, value, state=None):
		if type(reward) in [int, float] or (torch.is_tensor(reward)):
			self.historic_reward += reward if not torch.is_tensor(reward) else reward.item()
		self.history.append(self.event(reward, action, value, state))

	@property
	def length(self):
		return len(self.history)

class replay_buffer:
	def __init__(self):
		self.memory = deque(maxlen=2000)
		self.batch_size = 32

	def add(self, game):
		self.memory.append(game)

	def get_batch(self):
		if len(self.memory) < self.batch_size:
			return self.memory
		return random.sample(self.memory, self.batch_size)

	def is_full(self):
		return self.batch_size < len(self.memory)