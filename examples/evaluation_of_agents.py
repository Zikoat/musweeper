
from musweeper.agents import random_agent
from musweeper.agents.random_agent import *
from musweeper.evaluate_agent import *
import gym
import gym_minesweeper
from muzero_with_minesweeper import get_model as get_muzero_model

env = gym.make("Minesweeper-v0", width=10, height=10, mine_count=10)

#evaluate_agent(RandomAgent(env), env)
evaluate_agent(get_muzero_model(env)[0], env)


"""
BIG WORLD (10x10) and 10 mines

Radom agent stats
----
Wins: 0/10, 0%
Average reward: 1.4077777777777778
Median reward: 0.9222222222222223
Average steps: 4.5
Median steps: 3.5
Average unnecessary steps: 0.0

Best reward: 7.011111111111112
Best opened cells: 85/100
Avreage opened cells: 52.9/100
Median opened cells: 70.5/100
Highest steps: 13
---
Wins: 0/10, 0%
Average reward: 0.7522222222222222
Median reward: 0.8166666666666667
Average steps: 3.3
Median steps: 3.0
Average unnecessary steps: 0.0

Best reward: 2.3
Best opened cells: 76/100
Avreage opened cells: 45.0/100
Median opened cells: 61.5/100
Highest steps: 8
----
Average reward: 1.9144444444444448
Median reward: 0.9888888888888889
Average steps: 5.5
Median steps: 4.0
Average unnecessary steps: 0.0

Best reward: 6.811111111111112
Best opened cells: 84/100
Avreage opened cells: 46.5/100
Median opened cells: 63.0/100
Highest steps: 14

Muzero
----
Average reward: 2.4197111111111114
Median reward: 2.0722222222222224
Average steps: 5.9
Median steps: 4.5
Average unnecessary steps: 0.7

Best reward: 5.9207777777777775
Best opened cells: 82/100
Avreage opened cells: 62.3/100
Median opened cells: 74.0/100
Highest steps: 17

---
Wins: 0/10, 0%
Average reward: 2.205244444444445
Median reward: 1.1944444444444444
Average steps: 5.6
Median steps: 5.0
Average unnecessary steps: 0.9

Best reward: 8.809666666666667
Best opened cells: 85/100
Avreage opened cells: 52.0/100
Median opened cells: 67.5/100
Highest steps: 13

---


"""

"""
SMALL WORLD (5x5)

Random agent stats
----
Wins: 0/10, 0%
Average reward: 0.705
Average steps: 4.4
Average unnecessary steps: 0.0

Best reward: 1.95
Best opened cells: 20/25
Highest steps: 9
---
Wins: 0/10, 0%
Average reward: 0.8400000000000002
Average steps: 3.8
Average unnecessary steps: 0.0

Best reward: 2.1
Best opened cells: 17/25
Highest steps: 7
---
Wins: 0/10, 0%
Average reward: 0.74
Average steps: 2.7
Average unnecessary steps: 0.0

Best reward: 2.8
Best opened cells: 18/25
Highest steps: 8

Muzero
------
Average reward: 0.8150000000000001
Average steps: 4.1
Average unnecessary steps: 0.0

Best reward: 2.8999999999999995
Best opened cells: 19/25
Highest steps: 11
---
Wins: 0/10, 0%
Average reward: 0.99
Average steps: 4.3
Average unnecessary steps: 0.0

Best reward: 2.25
Best opened cells: 17/25
Highest steps: 10
---
Wins: 0/10, 0%
Average reward: 0.8450000000000001
Average steps: 4.1
Average unnecessary steps: 0.0

Best reward: 2.4
Best opened cells: 16/25
Highest steps: 10
"""
