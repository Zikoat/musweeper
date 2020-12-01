from unittest import TestCase

from src.example import basic_function


class TestBasicFunction(TestCase):
    def test_basic_function(self):
        assert basic_function() == 42
