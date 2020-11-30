import time

class clock:
	def __init__(self, timeout):
		self.start = time.time()
		self.end = self.start + timeout

	def __call__(self):
		return self.start < self.end

class monte_carlo_search_tree:
	def __init__(self, root, max_search_depth, random_rollout_metric=None):
		self.max_search_depth = max_search_depth
		self.timeout = 10

		self.root = root
		self.children_count = 10

		# coin-flip
		self.random_rollout_metric = (lambda tree, node: self.random.randint(0, 2) == 1) if random_rollout_metric is None else random_rollout_metric

	def expand(self):
		relative_depth_level = 0
		current_node = self.root

		# TODO : add "or" for if current node is a terminated state 
		while relative_depth_level < self.max_search_depth and clock(self.timeout):
			should_pick_random_node = self.random_rollout_metric(self, current_node)
			current_node_has_no_children = current_node.children is None
			# 
			if should_pick_random_node or current_node_has_no_children:
				# TODO : add something to invalidate invalid states
				current_node = current_node.get_a_children_node(self.children_count)
			else:		
				current_node = self.select(current_node.children)
			relative_depth_level += 1
		return current_node

		def select(self, nodes):
			return max(nodes, key=lambda x: x.score_metric())
