from gym.envs.registration import register

nondeterministic = False

register(
    id='MinesweeperGuided-v0',
    entry_point='src.envs.minesweeper_guided_env:MinesweeperGuidedEnv',
    nondeterministic=nondeterministic,
)

register(
    id='MinesweeperAssisted-v0',
    entry_point='src.envs.minesweeper_assisted_env:MinesweeperAssistedEnv',
    nondeterministic=nondeterministic,
)

# todo noflood
# todo Medium
# todo Expert
# todo GuidedExpert
# todo AssistedExpert
