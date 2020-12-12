import unittest

from .node import *
from .temperture import *

class TestTemperature(unittest.TestCase):
    def test_temperature(self):
        root = node(None)
        child_0 = root.create_node(0)
        child_1 = root.create_node(1)
        child_0.explored_count += 100
        child_1.explored_count += 200

        assert type(temperature_softmax(root)) in [int, np.int64]
        assert np.isclose(np.sum(create_distribution(root, 10)[0]), 1)
        assert np.isclose(np.sum(create_distribution(root, 50)[0]), 1)

if __name__ == "__main__":
    TestTemperature()