import tensorboard# as tb
from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_value(value, title):
	output_path = os.path.dirname(os.path.abspath(__file__))
	output_path = os.path.join(output_path, "output/")
	plt.clf()
	plt.title(title)
	plt.plot(value)
	plt.savefig(os.path.join(output_path, "{}.png".format(title.lower().replace(" ", "_"))))

if __name__ == "__main__" and len(sys.argv) == 2:
	experiment_file = sys.argv[1]
	tags = {
		
	}
	for summary in summary_iterator(experiment_file):
		for value_tag in summary.summary.value:
			tag, value = (value_tag.tag, value_tag.simple_value)
			if tag not in tags:
				tags[tag] = []
			tags[tag].append(value)
	if len(tags) == 0:
		raise Exception("empty tensorboard file")
	else:
		step = 40
		print(tags.keys())
		plot_value([
			np.mean(tags["game"][i:i + step]) for i in range(0, len(tags["game"]), step)
		], "Mean reward over time")
		plot_value([
			np.mean(tags["loss"][i:i + step]) for i in range(0, len(tags["loss"]), step)
		], "loss")
