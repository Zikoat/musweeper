import subprocess
import unittest
import gym
from ..envs.minesweeper_guided_env import MinesweeperGuidedEnv

# from minesweepr.minesweeper_util import generate_rules

class SolverException(Exception):
    pass

def api_solve(payload):
    try:
        return subprocess.run(
            [
                "C:/users/sscho/anaconda3/envs/mrgris/python.exe",
                "-c",
                "from minesweepr.minesweeper_util import api_solve;"+
                "print(api_solve({}))".format(payload)
            ],
            capture_output=True,
            check=True,
            timeout=1
        ).stdout
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
        assert not None in ob.shape[0]
        print()
        gym.spaces.box

    def test_call_api_solve(self):
        # todo use https://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout
        # and subprocess to run the
        rules = {"rules":[{"num_mines":1,"cells":["2-1"]},{"num_mines":0,"cells":["1-1"]},{"num_mines":0,"cells":[]}],"total_cells":2,"total_mines":1}
        result = api_solve("rules")

        print(rules)
        print(result)

        self.assertIn("'solution': {'1-1': 0.0, '2-1': 1.0}}", result)

    def test_solve_error(self):
        self.assertRaises(api_solve("shit"))

    def test_inconsistency(self):
        payload = {"rules":[{"num_mines":0,"cells":["1-1"]},{"num_mines":0,"cells":[]}],"total_cells":1,"total_mines":1}
        output = api_solve(payload).decode("utf-8")
        print(output)
        self.assertIsNone(output.solution, None)



if __name__ == '__main__':
    unittest.main()
