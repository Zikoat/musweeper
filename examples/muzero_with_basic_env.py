from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.model.muzero import *
from musweeper.muzero.utils.training_loop import *
import torch.optim as optim

env = BasicEnv()

representation, dynamics, prediction = create_model(env)
model = muzero(env, representation, dynamics, prediction, max_search_depth=3)
optimizer = optim.Adam(model.parameters())

print(model)

game_score = []
sum_loss = 0
for i in range(1000):
	if i % 100 == 0 and i > 0:
		print('%d: min=%.2f median=%.2f max=%.2f eval=%.2f, sum of 100 last games=%.2f, loss=%.2f' % (i, min(game_score), game_score[len(game_score)//2], max(game_score), sum(game_score)/len(game_score), sum(game_score[-100:]), sum_loss))
		sum_loss = 0
	model.reset()
	state = env.reset()
	done = False
	game_history = None
	sum_score = 0
	while not done:
		output = model.plan_action(state)
		best_node = max(output, key=lambda x: x.value)
		best_action = best_node.node_id
		best_value = best_node.value

		game_history = model.tree.get_rollout_path(game_state=game_history)
		state, reward, done = env.step(best_action)

		model.update(state, best_action)
		game_history.actual_reward.append(torch.tensor([reward]).reshape((1, -1)))
		game_history.state.append(state.reshape((1, -1)))
		game_history.actual_action.append(best_action)
		game_history.actual_value.append(best_value)
		sum_score += reward

	optimizer.zero_grad()
	total_loss = loss_from_game(model, game_history)
	total_loss.backward()
	optimizer.step()
	game_score.append(sum_score)
	sum_loss += total_loss.item()