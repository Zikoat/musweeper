import unittest
import pytest
from ..envs.mrgris.mrgris_python3 import api_solve


def test_api_solve_with_board():
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
    result = api_solve(board)
    print(result)
    assert result["processing_time"] >= 0
    assert result["solution"] == {  # note: the coordinates are given as 'y-x'
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


def test_inconsistency():
    payload = {"rules": [{"num_mines": 0, "cells": ["1-1"]},
                         {"num_mines": 0, "cells": []}],
               "total_cells": 1,
               "total_mines": 1}
    output = api_solve(payload)
    print(output)
    print("is of type", type(output))
    assert output["solution"] is None


def test_solve_error():
    with pytest.raises(TypeError):
        api_solve("wrong input")


def test_solve_with_rules():
    rules = {"rules": [{"num_mines": 1, "cells": ["2-1"]},
                       {"num_mines": 0, "cells": ["1-1"]},
                       {"num_mines": 0, "cells": []}],
             "total_cells": 2,
             "total_mines": 1}
    result = api_solve(rules)

    print(rules)
    print(result)
    print("is of type", type(result))

    assert {'1-1': 0.0, '2-1': 1.0} == result["solution"], \
        "The solution returned from the solver is incorrect"
    assert result["processing_time"] >= 0, "The processing time is wrong"
