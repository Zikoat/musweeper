import unittest
import gym


class MinesweeperGuidedEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("MinesweeperGuided-v0")

    def setUp(self) -> None:
        self.env.reset()

    def test_has_correct_dimensions(self):
        ob, reward, episode_over, info = self.env.step(self.env.action_space.sample())
        self.env.reset()
        self.assertTupleEqual(self.env.observation_space.shape, (2, 8, 8))
        self.assertTupleEqual(ob.shape, (2, 8, 8))
        assert not None in ob.shape[0]
        print()
        gym.spaces.box


if __name__ == '__main__':
    unittest.main()
