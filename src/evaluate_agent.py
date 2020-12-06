from gym import wrappers, logger
import numpy as np

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
    print("Evaluating the agent {} on the environment {} for {} episodes".format(type(agent).__name__, type(env).__name__, episode_count))

    for i in range(episode_count):

        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                stats.append([reward, *_.values()])
                break
    env.close()

    stats = np.array(stats, dtype="object")
    reward = stats[:,0]
    opened_cells = stats[:,1]
    steps = stats[:,2]
    unnecessary_steps = stats[:,3]
    game_over = stats[:,4]

    print("Wins: {}/{}, {}%".format(np.count_nonzero(game_over)-episode_count, episode_count, (1 - np.average(game_over)) * 100))
    print("Average reward: {}".format(np.mean(reward)))
    print("Average steps: {}".format(np.mean(steps)))
    print("Average unnecessary steps: {}".format(np.mean(unnecessary_steps)))
    print("")
    print("Best reward: {}".format(np.max(reward)))
    print("Best opened cells: {}/54".format(np.max(opened_cells)))
    print("Highest steps: {}".format(np.max(steps)))
