

class game_history:
	def __init__(self):
		self.reward = []
		self.actual_reward = []

		self.state = []
		self.action = []

		self.value = []
		self.policy = []

	@property
	def length(self):
		return len(self.state)
