
def self_play_single_player(model, env, games=10):
    """
    Since the model will is playing a single player game
    the self play algorithm needs to be adjusted to account for this

    Parameters
    ----------
    model : muzero
        muzero - since we are working with only one model (since it's single player) no need for model storage
    env : gym.Env
        the environment where the model will be playing
    """
    history = []
    for _ in range(games):
        history.append(play_game())
    sorted_history = sorted(history, lambda x: x.historic_reward)
    middle = len(sorted_history) // 2
    bad_plays = sorted_history[:middle]
    win_plays = sorted_history[middle:]
    loss = 0
    for player, sign in zip([bad_plays, win_plays], [-1, 1]):
        for game in bad_plays:
            loss += torch.sum([
                sign * event.policy for event in game 
            ])
    return loss
    