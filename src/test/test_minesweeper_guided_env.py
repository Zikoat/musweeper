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
            timeout=2
        ).stdout.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        raise SolverException("api_solve errored with message below:\n\n{}"
                              .format(e.stderr.decode("utf-8")))


def parse_coordinate(coordinate):
    y, x = list(map(int, coordinate.split("-")))
    return x, y


def get_probability_matrix(result, width, height):
    if "_other" in result:
        matrix = np.array((width, height), result["_other"])
    else:
        matrix = np.empty((width, height))

    for coordinate_string, mine_probability in result["solution"].items():
        x, y = parse_coordinate(coordinate_string)
        matrix[x-1, y-1] = mine_probability
    assert not None in matrix
    return matrix


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

    def test_solve_with_rules(self):
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

    def test_solve_with_board(self):
        # This test is based on the example given in
        # https://github.com/mrgriscom/minesweepr/blob/master/README.md
        board = {"board":"""..1xxxxxxx
..2xxxxxxx
..3xxxxxxx
..2xxxxxxx
112xxxxxxx
xxxxxxxxxx
xxxxxxxxxx
xxxxxxxxxx
xxxxxxxxxx
xxxxxxxxxx""",
            "total_mines": 10}
        result = api_solve(board)
        print(result)
        self.assertGreaterEqual(result["processing_time"], 0)

        self.assertEqual(result["solution"],
                         {  # note: the coordinates are given as 'y-x'
                             '01-01': 0.0,
                             '01-02': 0.0,
                             '01-03': 0.0,
                             '01-04': 0.0, # A
                             '02-01': 0.0,
                             '02-02': 0.0,
                             '02-03': 0.0,
                             '02-04': 1.0, # B
                             '03-01': 0.0,
                             '03-02': 0.0,
                             '03-03': 0.0,
                             '03-04': 1.0, # C
                             '04-01': 0.0,
                             '04-02': 0.0,
                             '04-03': 0.0,
                             '04-04': 1.0, # D
                             '05-01': 0.0,
                             '05-02': 0.0,
                             '05-03': 0.0,
                             '05-04': 0.0, # E
                             '06-01': 0.07792207792207793, # I
                             '06-02': 0.9220779220779222, # H
                             '06-03': 0.0, # G
                             '06-04': 0.07792207792207793, # F
                             '_other': 0.07792207792207792, # None
                         })

    def test_create_probability_matrix_from_solution(self):
        board = "x11\nxxx"
        total_mines = 1
        result = api_solve({"board": board, "total_mines": total_mines})
        width = 3
        height = 2

        probability_matrix = get_probability_matrix(result, width, height)
        expected_probability_matrix = np.array([[0, 0], [0, 0.5], [0, 0.5]])

        np.testing.assert_array_equal(expected_probability_matrix, probability_matrix)




if __name__ == '__main__':
    unittest.main()
