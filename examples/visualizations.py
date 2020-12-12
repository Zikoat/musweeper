import time

import torch.optim as optim
from musweeper.muzero.utils.clock import clock
from musweeper.muzero.utils.training_loop import train
from musweeper.muzero.model.components import transform_input
from musweeper.muzero.model.muzero import *
from musweeper.muzero.model.selfplay import play_game, selfplay_single_player
from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.utils.training_loop import *

env = BasicEnv(state_size=2)
representation, dynamics, prediction = create_model(env)
model = muzero(env, representation, dynamics, prediction, max_search_depth=2)

model.plan_action(env.reset())
tree = model.tree
#tree = model.init_tree(env.reset())
#tree.expand_node(1, model, max_leaf_node_count=1)
#tree.expand_node(1, model, max_leaf_node_count=1)
#tree.expand_node(1, model, max_leaf_node_count=1)
#tree.expand_node(1, model, max_leaf_node_count=1)
tree.draw(file_name="first", show_only_used_edges=True)

#tree.update_root(None, 1)
#tree.expand_node(1, model, max_leaf_node_count=1)
#tree.draw(file_name="second")
