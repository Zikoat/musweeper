from collections import deque
import random
from collections import namedtuple
import torch

class game_event_history:
	def __init__(self):
		self.history = []
		self.event = namedtuple('event', ['reward', 'action', 'value', 'state', 'policy', 'info'])
		self.historic_reward = 0

	def add(self, reward, action, value, policy=None, state=None, info=None, soft=False):	
		if type(reward) in [int, float] or (torch.is_tensor(reward)):
			self.historic_reward += reward if not torch.is_tensor(reward) else reward.item()
		event = self.event(reward, action, value, state, policy, info)
		if soft:
			return event
		self.history.append(event)
	
	@property
	def length(self):
		"""
		Length of the game

		Returns
		-------
		int
			game length
		"""
		return len(self.history)

class replay_buffer:
	def __init__(self):
		self.memory = deque(maxlen=512)
		self.batch_size = 8

	def add(self, game):
		"""
		Add game to replay buffer

		Parameters
		----------
		game : game_event_history
			the game event history
		"""
		self.memory.append(game)

	def get_batch(self):
		"""
		Get a batch from replay buffer

		Returns
		-------
		list
			random sample of the memory with the given batch-size 
		"""
		if len(self.memory) < self.batch_size:
			return self.memory
		return random.sample(self.memory, self.batch_size)

	def is_full(self):
		"""
		Check if the replay buffer is full

		Returns
		-------
		bool
			is the the replay buffer is big enough for a batch
		"""
		return self.batch_size < len(self.memory)
