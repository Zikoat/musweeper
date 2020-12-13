from musweeper.musweeper.muzero.model.components import BasicConv
import unittest
import torch

class TestComponents(unittest.TestCase):
    def test_conv(self):
        component = BasicConv()#1, 4, hidden_output=16)
        env = torch.rand(1, 1, 5, 5)
        output = component(env)
        assert output.shape[0] == 1
        assert output.shape[1] == 120
        assert output.dim() == 2

if __name__ == "__main__":
    TestComponents()