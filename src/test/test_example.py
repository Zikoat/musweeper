import unittest
from src.example import basic_function


class TestExample(unittest.TestCase):
    def test_upper(self):
        self.assertEqual(basic_function(), 42)


if __name__ == '__main__':
    unittest.main()
