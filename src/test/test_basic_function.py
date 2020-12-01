from unittest import TestCase

from src.example import basic_function


class TestBasicFunction(TestCase):
    def test_basic_function(self):
        print("wow")
        assert basic_function() == 42

