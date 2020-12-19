import unittest
import gym
import numpy as np


class MinesweeperGuidedEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("MinesweeperGuided-v0")

    def setUp(self) -> None:
        self.env.reset()

    def test_has_correct_dimensions(self):
        ob, reward, episode_over, info = self.env.step(
            self.env.action_space.sample())
        self.env.reset()
        self.assertTupleEqual(self.env.observation_space.shape, (2, 8, 8))
        self.assertTupleEqual(ob.shape, (2, 8, 8))
        self.assertNotIn(None, ob)

    def test_create_probability_matrix_from_solution(self):
        env = gym.make("MinesweeperGuided-v0", width=3, height=4, mine_count=3)
        env.reset()

        # Remove all the mines
        env.mines = np.zeros((env.width, env.height))
        # Plant mines
        env.mines[2, 2] = 1
        env.mines[0, 1] = 1

        # Open top middle cell
        env.step(0)

        # Open top right cell
        ob, reward, episode_over, info = env.step(2)
        print(env.render("ansi"))
        print(ob)
        self.assertEqual(env.render("ansi"), "11.\nx21\nxxx\nxxx")

        probability_matrix = env.get_probability_matrix()
        expected_probability_matrix = np.array([
            [0, 1, 0,   1/3],
            [0, 0, 1/2, 1/3],
            [0, 0, 1/2, 1/3],
        ])

        np.testing.assert_array_equal(expected_probability_matrix,
                                      probability_matrix)

    def test_reset(self):
        reset_obs = self.env.reset()
        step_obs, *step_other_state = self.env.step(self.env.action_space.sample())

        self.assertEqual(reset_obs.shape, step_obs.shape)
        self.assertEqual(reset_obs.size, step_obs.size)
        self.assertEqual(type(reset_obs), type(step_obs))

if __name__ == '__main__':
    unittest.main()
