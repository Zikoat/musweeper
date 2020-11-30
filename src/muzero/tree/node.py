import numpy as np
import random

class node:
	def __init__(self, parrent, node_id=None):
		self.children = None
		self.node_id = node_id
		self.parrent = parrent
		
		self.value = 0
		self.explored_count = 0
		self.wins_count = 0

		self.depth = 0 if parrent is None else (parrent.depth + 1)

		self.avaible_children_paths = None
		self.score_metric = self.search_value_exploration_explotation

	def search_value_exploration_explotation(self):
		# https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
		parrent_explored = np.log2(self.parrent.explored_count)/self.explored_count if self.parrent.explored_count != 1 and self.explored_count != 0 else 0
		child_explored = self.wins_count / self.explored_count if self.explored_count > 0 else 0
		c = np.sqrt(2)

		return child_explored + c * np.sqrt(parrent_explored)

	def get_a_children_node(self, children_count):
		if self.avaible_children_paths is None:
			self.avaible_children_paths = list(range(children_count))
		picked_node = self.avaible_children_paths[random.randint(0, len(self.avaible_children_paths) - 1)]
		self.avaible_children_paths.remove(picked_node)
		return node(self, node_id=picked_node)

	def __str__(self):
		return "{} {} {}".format(self.node_id, self.value, self.depth)

	def __repr__(self):
		return self.__str__()
