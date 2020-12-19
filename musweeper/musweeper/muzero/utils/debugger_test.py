import unittest
from .debugger import *


class TestDebugger(unittest.TestCase):
    def test_model_debugger(self):
        first_instance = model_debugger()
        first_instance.start_forward("test")
        first_instance.add_element("test", "swag")
        first_instance.stop_forward("test")

        second_instance = model_debugger()

        assert first_instance.model_forward_data == second_instance.model_forward_data
        assert len(first_instance.get_last_round()) == 1
        assert len(list(first_instance.get_last_round())[0]) == 2
        for key, value in first_instance.get_last_round():
            assert key == "test"
            assert value == ["swag"]


if __name__ == '__main__':
    unittest.main()
