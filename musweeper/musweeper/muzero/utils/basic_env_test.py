import unittest
import torch
from .basic_env import *


class TestBasicEnv(unittest.TestCase):
    def test_basic_env(self):
        env = BasicEnv()
        assert torch.is_tensor(env.reset())
        for i in range(env.timeout):
            state, reward, done = env.step((i + 1) % env.action_size)
            assert reward == 1
            assert state[(i + 1) % env.action_size].item() == -1
            assert state[(i) % env.action_size].item() == 1
            assert not done

    def test_super_basic(self):
        env = BasicEnv(super_simple=True)
        assert torch.is_tensor(env.reset())
        for i in range(env.timeout):
            state, reward, done = env.step((i + 1) % env.action_size)
            assert reward in [0, 1]
            assert not done
            
if __name__ == '__main__':
    unittest.main()
