from gym import wrappers, logger
import numpy as np

def victory(env):
    arr = env.open_cells.flatten()
    sum_arr = arr[arr == -1]
    return sum_arr.shape == env.mines_count and not env._game_over()

def evaluate_agent(agent, env):
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    outdir = '/tmp/random-agent-results'

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    episode_count = 10
    reward = 0
    done = False
    stats = []
    print("Evaluating the agent {} on the environment {} for {} episodes"
          .format(type(agent).__name__, type(env).__name__, episode_count))

    sum_reward = 0
    for i in range(episode_count):
        ob = env.reset()
        sum_reward = 0
        while True:
            action = agent.act(ob, env)
            ob, reward, done, info = env.step(action)
            sum_reward += reward
            if done:
                stats.append([sum_reward, *info.values()])
                print(victory(env))
                break
    env.close()

    stats = np.array(stats, dtype="object")
    reward = stats[:, 0]
    opened_cells = stats[:, 1]
    steps = stats[:, 2]
    unnecessary_steps = stats[:, 3]
    game_over = stats[:, 4]

    print("Wins: {}/{}, {}%".format(np.count_nonzero(game_over) - episode_count,
                                    episode_count,
                                    (1 - np.average(game_over)) * 100))
    print("Average reward: {}".format(np.mean(reward)))
    print("Median reward: {}".format(np.median(reward)))
    print("Average steps: {}".format(np.mean(steps)))
    print("Median steps: {}".format(np.median(steps)))
    print("Average unnecessary steps: {}".format(np.mean(unnecessary_steps)))
    print("")
    print("Best reward: {}".format(np.max(reward)))
    print("Best opened cells: {}/{}".format(np.max(opened_cells), env.width * env.height))
    print("Avreage opened cells: {}/{}".format(np.mean(opened_cells), env.width * env.height))
    print("Median opened cells: {}/{}".format(np.median(opened_cells), env.width * env.height))
    print("Highest steps: {}".format(np.max(steps)))

    return stats
