import tensorboard# as tb
from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np
import matplotlib.pyplot as plt

experiment_id = "/Users/2xic/NTNU/musweepr/musweeper/ignore/runs/Dec18_21-24-52_861aaa6f6fe0/events.out.tfevents.1608326692.861aaa6f6fe0.6.1"

tags = {
	
}
for summary in summary_iterator(experiment_id):
	for v in summary.summary.value:
		tag, value = (v.tag, v.simple_value)
		if tag not in tags:
			tags[tag] = []
		tags[tag].append(value)
print(tags.keys())
#plt.plot(np.log(tags["loss"]))
#plt.show()

plt.plot(np.log(tags["avg_score"]))
plt.show()
