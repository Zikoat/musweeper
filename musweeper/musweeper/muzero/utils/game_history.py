

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
