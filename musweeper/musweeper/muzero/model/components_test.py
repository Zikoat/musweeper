from musweeper.musweeper.muzero.model.components import BasicConv
import unittest
import torch

class TestComponents(unittest.TestCase):
    def test_conv(self):
        component = BasicConv()
        env = torch.rand(1, 1, 10, 10)
        output = component(env)
        assert output.shape[0] == 1
        assert output.shape[1] == 120
        assert output.dim() == 2

if __name__ == "__main__":
    TestComponents()
    