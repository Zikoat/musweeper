from gym.envs.registration import register

register(
    id='MinesweeperGuided-v0',
    entry_point='src.envs.minesweeper_guided_env:MinesweeperGuidedEnv',
    nondeterministic=False,
)
