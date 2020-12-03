from gym.envs.registration import register

register(
    id='MinesweeperGuided-v0',
    entry_point='gym_soccer.envs:MinesweeperGuidedEnv',
    nondeterministic=False,
)
