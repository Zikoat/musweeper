import time

def singleton(cls):
	obj = cls()
	# Always return the same object
	cls.__new__ = staticmethod(lambda cls: obj)
	# Disable __init__
	try:
		del cls.__init__
	except AttributeError:
		pass
	return cls

@singleton
class model_debugger:
	def __init__(self):
		self.data = {

		}
		self.active_forwards = {

		}
		self.logs = [

		]

	def get_last_round(self):
		return self.data.items()

	def start_forward(self, name):
		if name not in self.active_forwards:
			self.active_forwards[name] = []
	
	def add_element(self, name, element):
		self.active_forwards[name].append(element)
	
	def stop_forward(self, name):
		self.data[name] = self.active_forwards[name]
		self.active_forwards[name] = []

	def log(self, value):
		self.logs.append(value)

class debugger:
	def __init__(self):
		self.time_usage = {

		}
		self.active_timers = {

		}

		self.values_to_plot = [

		]
	
	def start_track_time(self, name):   
		if name not in self.active_timers:
			self.active_timers[name] = []
			self.time_usage[name] = 0
		self.active_timers[name].append(time.time())

	def stop_track_time(self, name):
		time_usage = time.time() - self.active_timers[name].pop(0)
		self.time_usage[name] += time_usage
	
	def track_values(self, values):
		"""
		Used for tracking values
		loss, reward over time, etc

		Parameters
		----------
		values : [type]
			[description]
		"""
		self.values_to_plot.append(values)