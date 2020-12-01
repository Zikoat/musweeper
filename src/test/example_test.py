import unittest
from example import basic_fucntion

class TestExample(unittest.TestCase):
	def test_upper(self):
		self.assertEqual(basic_fucntion(), 42)

if __name__ == '__main__':
	unittest.main()

