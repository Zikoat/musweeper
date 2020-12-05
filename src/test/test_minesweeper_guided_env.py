import subprocess
import unittest
import gym
from ..envs.minesweeper_guided_env import MinesweeperGuidedEnv
import ast
import numpy as np

# from minesweepr.minesweeper_util import generate_rules

class SolverException(Exception):
    pass

def api_solve(payload):
    try:
        return ast.literal_eval(subprocess.run(
            [
                "C:/users/sscho/anaconda3/envs/mrgris/python.exe",
                "-c",
                "from minesweepr.minesweeper_util import api_solve;"+
                "print(api_solve({}))".format(payload)
            ],
            capture_output=True,
            check=True,
            timeout=1
        ).stdout.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        raise SolverException("api_solve errored with message below:\n\n{}"
                              .format(e.stderr.decode("utf-8")))


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
        self.assertNotIn(None, ob)

    def test_call_api_solve(self):
        rules = {"rules":[{"num_mines":1,"cells":["2-1"]},{"num_mines":0,"cells":["1-1"]},{"num_mines":0,"cells":[]}],"total_cells":2,"total_mines":1}
        result = api_solve(rules)

        print(rules)
        print(result)
        print("is of type", type(result))

        self.assertEqual({'1-1': 0.0, '2-1': 1.0}, result["solution"], "The solution returned from the solver is incorrect")
        self.assertGreaterEqual(result["processing_time"], 0, "The processing time is wrong")

    def test_solve_error(self):
        self.assertRaises(SolverException, lambda: api_solve("wrong input"))

    def test_inconsistency(self):
        payload = {"rules":[{"num_mines":0,"cells":["1-1"]},{"num_mines":0,"cells":[]}],"total_cells":1,"total_mines":1}
        output = api_solve(payload)
        print(output)
        print("is of type", type(output))
        self.assertIsNone(output["solution"], None)


if __name__ == '__main__':
    unittest.main()
