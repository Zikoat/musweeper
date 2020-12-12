import unittest
import gym
import numpy as np

from src.envs.minesweeper_guided_env import SolverException


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

    def test_solve_with_rules(self):
        rules = {"rules": [{"num_mines": 1, "cells": ["2-1"]},
                           {"num_mines": 0, "cells": ["1-1"]},
                           {"num_mines": 0, "cells": []}],
                 "total_cells": 2,
                 "total_mines": 1}
        result = self.env.api_solve(rules)

        print(rules)
        print(result)
        print("is of type", type(result))

        self.assertEqual({'1-1': 0.0, '2-1': 1.0}, result["solution"],
                         "The solution returned from the solver is incorrect")
        self.assertGreaterEqual(result["processing_time"], 0,
                                "The processing time is wrong")

    def test_solve_error(self):
        self.assertRaises(SolverException,
                          lambda: self.env.api_solve("wrong input"))

    def test_inconsistency(self):
        payload = {"rules": [{"num_mines": 0, "cells": ["1-1"]},
                             {"num_mines": 0, "cells": []}],
                   "total_cells": 1,
                   "total_mines": 1}
        output = self.env.api_solve(payload)
        print(output)
        print("is of type", type(output))
        self.assertIsNone(output["solution"], None)

    def test_api_solve_with_board(self):
        # This test is based on the example given in
        # https://github.com/mrgriscom/minesweepr/blob/master/README.md
        board = {"board": "..1xxxxxxx\n"
                          "..2xxxxxxx\n"
                          "..3xxxxxxx\n"
                          "..2xxxxxxx\n"
                          "112xxxxxxx\n"
                          "xxxxxxxxxx\n"
                          "xxxxxxxxxx\n"
                          "xxxxxxxxxx\n"
                          "xxxxxxxxxx\n"
                          "xxxxxxxxxx",
                 "total_mines": 10}
        result = self.env.api_solve(board)
        print(result)
        self.assertGreaterEqual(result["processing_time"], 0)

        self.assertEqual(
            result["solution"],
            {  # note: the coordinates are given as 'y-x'
                '01-01': 0.0,
                '01-02': 0.0,
                '01-03': 0.0,
                '01-04': 0.0,  # A
                '02-01': 0.0,
                '02-02': 0.0,
                '02-03': 0.0,
                '02-04': 1.0,  # B
                '03-01': 0.0,
                '03-02': 0.0,
                '03-03': 0.0,
                '03-04': 1.0,  # C
                '04-01': 0.0,
                '04-02': 0.0,
                '04-03': 0.0,
                '04-04': 1.0,  # D
                '05-01': 0.0,
                '05-02': 0.0,
                '05-03': 0.0,
                '05-04': 0.0,  # E
                '06-01': 0.07792207792207793,  # I
                '06-02': 0.9220779220779222,  # H
                '06-03': 0.0,  # G
                '06-04': 0.07792207792207793,  # F
                '_other': 0.07792207792207792,  # None
            }
        )

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
