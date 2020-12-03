from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.model.muzero import *
from musweeper.muzero.utils.training_loop import *
import torch.optim as optim

env = BasicEnv()

representation, dynamics, prediction = create_model(env)
model = muzero(env, representation, dynamics, prediction, max_search_depth=3)
optimizer = optim.Adam(model.parameters())

game_score = []
for i in range(1000):
	if i % 100 == 0 and i > 0:
		print('%d: min=%.2f median=%.2f max=%.2f eval=%.2f' % (i, min(game_score), game_score[len(game_score)//2], max(game_score), sum(game_score)/len(game_score)))
	model.reset()
	state = env.reset()
	done = False
	game_history = None
	sum_score = 0
	while not done:
		output = model.plan_action(state)
		best_action = max(output, key=lambda x: x.value).node_id

		game_history = model.tree.get_rollout_path(seed_state=model.tree.root.environment_state, game_state=game_history)
		state, reward, done = env.step(best_action)

		model.update(state, best_action)
		game_history.actual_reward.append(torch.tensor([reward]).reshape((1, -1)))
		game_history.state.append(state.reshape((1, -1)))
		game_history.action.append(best_action)
		sum_score += reward

	optimizer.zero_grad()
	total_loss = loss_from_game(model, game_history)
	total_loss.backward()
	optimizer.step()
	game_score.append(sum_score)
