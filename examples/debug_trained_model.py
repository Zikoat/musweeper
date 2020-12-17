import time
import gym
import gym_minesweeper
import torch.optim as optim
from musweeper.muzero.utils.clock import clock
from musweeper.muzero.utils.training_loop import train
from musweeper.muzero.model.components import transform_input
from musweeper.muzero.model.muzero import *
from musweeper.muzero.model.selfplay import play_game, selfplay_single_player
from musweeper.muzero.utils.basic_env import *
from musweeper.muzero.utils.training_loop import *
from musweeper.muzero.tree.temperture import *

def print_parameters():
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)#, param.data

#env = BasicEnv(state_size=2)#, super_simple=True)

env = gym.make("Minesweeper-v0", width=5, height=5, mine_count=5)
representation, dynamics, prediction = create_model(env)
model = muzero(env, representation, dynamics, prediction, max_search_depth=2)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

model.load("../ignore/muzero", optimizer, cpu=True)

state = env.reset()
#for _ in range(3):
done = False
while not done: 
    print(env.render("ansi"))
#    print(state)
#    print(model.prediction(model.representation(state)))
    distro, _ = create_distribution(model.plan_action(state), T=1)
#    print(distro)
    best_action = get_action(distro)
    state, reward, done = env.step(best_action)[:3]
    print(best_action, reward, done, env._parse_action(best_action))
    if done:
        print(env.render("ansi"))
#        print(state)
"""
print(model.prediction(model.representation(torch.tensor([1, 0, 0, 0])[:env.action_size])))
print(model.prediction(model.representation(torch.tensor([0, 1, 0, 0])[:env.action_size])))
print(model.dynamics(model.representation(torch.tensor([0, 1, 0, 0])[:env.action_size]), 0))
print(model.dynamics(model.representation(torch.tensor([0, 1, 0, 0])[:env.action_size]), 1))

print(temperature_softmax(model.plan_action(torch.tensor([0, 1, 0, 0])[:env.action_size])))
print(temperature_softmax(model.plan_action(torch.tensor([1, 0, 0, 0])[:env.action_size])))

model.reset()
print(create_distribution(model.plan_action(torch.tensor([0, 1, 0, 0])[:env.action_size]), T=1))
model.reset()
print(create_distribution(model.plan_action(torch.tensor([1, 0, 0, 0])[:env.action_size]), T=1))
"""

#print(list(model.parameters()))
#model.plan_action(torch.tensor([1, 0, 0, 0]))
#model.tree.draw("model")
