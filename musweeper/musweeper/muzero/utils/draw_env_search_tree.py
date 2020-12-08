
def see_model_search_tree(model, env):
	model.reset()
	done = False
	observation = env.reset()
	original_root = None
	while not done:
		output = model.plan_action(observation)
		if original_root is None:
			original_root = model.tree.root
		best_node = max(output, key=lambda node: node.value)
		best_action = best_node.node_id
		observation, reward, done = env.step(best_action)[:3]
		model.update(observation, best_action)
	return [model.tree.draw(), model.tree.draw(show_only_used_edges=True)]
