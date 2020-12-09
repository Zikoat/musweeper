import random
import tempfile
import time

import torch
from graphviz import Digraph

from ..utils.game_history import *
from .node import *


class monte_carlo_search_tree:
    def __init__(self, root_state, max_search_depth, action_size=2, random_rollout_metric=None, top_k_nodes_to_search=5, add_exploration_noise=True):
        self.max_search_depth = max_search_depth
        self.timeout = 10
        assert root_state is None or torch.is_tensor(
            root_state), " {} {}".format(root_state, type(root_state))
        self.root = node(parrent=None, hidden_state=root_state)
        self.originale_root = self.root
        self.children_count = action_size
        self.add_exploration_noise = add_exploration_noise

        # coin-flip
        self.random_rollout_metric = (lambda tree, node: random.randint(
            0, 2) == 1) if random_rollout_metric is None else random_rollout_metric
        self.top_k_nodes_to_search = top_k_nodes_to_search

    def select_best_node(self, node, model):
        """
        Select best child node

        Parameters
        ----------
        node : node
            node to exploit

        Returns
        -------
        node
            best node
        """
        found_leaf = False
        if len(node.children) == 0:
            for action in range(self.children_count):
                self.set_values_for_expand_a_node(node.create_node(action), model)
            found_leaf = True
        best_child_node = max(list(node.children.values()), key=lambda node: node.upper_confidence_boundary())
        return best_child_node, found_leaf

    def expand_node(self, node, model, max_leaf_node_count=100):
        """
        Expand node

        Parameters
        ----------
        node : node
                seed node
        model : muzero
                model used for predictions of reward

        Returns
        -------
        node
                the leaf node
        """
        if type(node) == int:
            node = self.root.create_children_if_not_exist(node)

        if model and not node.has_init:
            self.set_values_for_expand_a_node(node, model)
        
        leaf_node_count = 0
        while leaf_node_count < max_leaf_node_count:
            self.backpropgate(self.expand(node, model), start_depth=node.depth)
            leaf_node_count += 1
    
    def expand(self, current_node=None, model=None):
        """
        Expand the search tree

        Parameters
        ----------
        current_node : node, optional
                the seed node, by default None
        model : model, optional
                model used to update the reward from a given model, by default None

        Returns
        -------
        node
                leaf node after search
        """
        relative_depth_level = 0
        current_node = self.root if current_node is None else current_node
        found_leaf = False
        if self.add_exploration_noise:
            current_node.add_exploration_noise()
        while relative_depth_level < self.max_search_depth and not found_leaf:
            current_node, found_leaf = self.select_best_node(current_node, model)
            relative_depth_level += 1
        return current_node

    def set_values_for_expand_a_node(self, new_node, model, is_child_node=False):
        # when a new node is found, we assign reward and policy from the model (see Expansion in Appendix B for details)
        hidden_state = new_node.parrent.hidden_state
        action = new_node.node_id

        if not new_node.has_init:
            state, action_tensor = hidden_state, torch.tensor([action]).float()
            next_state, reward = model.dynamics(state, action_tensor)
            policy, value_function = model.prediction(state)
            new_node.on_node_creation(
                next_state, reward, policy, value_function)
        else:
            next_state = new_node.hidden_state

        # create the child actions from the current node and assign a prior based on model output
        if not is_child_node:
            next_state_policy, _ = model.prediction(next_state)
            next_state_policy = next_state_policy[0] if len(
                next_state_policy.shape) > 1 else next_state_policy
            nodes_to_search = min(
                next_state_policy.shape[-1], self.top_k_nodes_to_search)
            _, indices = torch.topk(next_state_policy, nodes_to_search)
            for index in range(nodes_to_search):  # self.children_count):
                action = indices[index].item()
                new_node.children[action] = node(
                    parrent=new_node, node_id=action, hidden_state=None, prior=next_state_policy[action].item())
                self.set_values_for_expand_a_node(
                    new_node.children[action], model, is_child_node=True)

    def backpropgate(self, leaf_node, start_depth, discount=0.1):
        """
        When a leaf node is found, the values will be backpropgated and updated upwards
        """
        cumulative_discounted_reward = 0
        visited_nodes = 0
        value = leaf_node.value_of_model.item() if torch.is_tensor(
            leaf_node.value_of_model) else leaf_node.value_of_model
        assert type(value) in [
            int, float], "value should be defined correctly {} {}".format(value, type(value))

        depth = (leaf_node.depth - start_depth)
        while leaf_node is not None and leaf_node != self.root:
            node_level_diff = depth - visited_nodes
#			cumulative_discounted_reward += (discount ** (node_level_diff) * leaf_node.reward) + leaf_node.value_of_model * discount ** visited_nodes

            leaf_node.value += value
            leaf_node.explored_count += 1
            leaf_node.min_max_node_tracker.update(leaf_node.node_value())
            visited_nodes += 1

            reward = leaf_node.reward.item() if torch.is_tensor(
                leaf_node.reward) else leaf_node.reward
            leaf_value = leaf_node.value_of_model.item() if torch.is_tensor(
                leaf_node.value_of_model) else leaf_node.value_of_model
            assert type(value) in [
                int, float], "value should be defined correctly {} {}".format(value, type(value))
            assert type(reward) in [
                int, float], "reward should be defined correctly {}".format(reward)
            discounted_reward = (discount ** (node_level_diff) * reward)
            discounted_value = leaf_value * discount ** visited_nodes
            # TODO : ASSUMING A 0 TO 1 REWARD
            assert 0 <= discounted_reward <= 1
            assert 0 <= discounted_value <= 1
            value += discounted_reward + discounted_value

            leaf_node.cumulative_discounted_reward = cumulative_discounted_reward
            leaf_node = leaf_node.parrent

        return value  # cumulative_discounted_reward



    def update_root(self, state, action):
        """
        Update root for when a action is taken

        Parameters
        ----------
        state : torch
                the new environment state
        action : int
                the action taken.
        """
        assert type(action) == int, "action should be int"
        self.root = self.root.children[action]
        self.root.environment_state = state

    def get_rollout_path(self, game_state=None):
        """
        Get the rollout path taken for the tree

        Parameters
        ----------
        game_state : game_history, optional
                used if you want to append to history, by default None

        Returns
        -------
        game_history
                the game history from the search tree
        """
        current_node = self.root
        game_state = game_event_history() if game_state is None else game_state
        while 0 < len(current_node.children.keys()):
            best_node = max(current_node.children.items(),
                            key=lambda x: x[1].value)[1]
            game_state.add(
                reward=best_node.reward,
                action=best_node.policy,
                value=best_node.value
            )
            current_node = best_node
        return game_state

    def construct_tree(self, dot, node, show_only_used_edges):
        if node is None:
            return None

        def create_node(node, create=True):
            depth, action = node.depth, node.node_id
            if action is None:
                action = "root"
            state = node.environment_state if node.environment_state is not None else None
            score = node.score_metric()
            parrent_action = node.parrent.node_id if node.parrent is not None else "seed"

            node_id = '{depth}_{parrent_action}_{action}_{state}'.format(
                depth=depth, parrent_action=parrent_action, action=action, state=state)
            if create:
                ucb = node.upper_confidence_boundary()
                ucb_reason = node.ucb_score_parts
                dot.node(node_id, 'depth {depth}, action {action}, score {score}, env state: {state}, ucb = {ucb}, ucb_reason = {ucb_reason}'.format(
                    depth=depth, action=action, score=score, state=state, ucb=ucb, ucb_reason=ucb_reason), color='black' if state is None else 'green')
            return node_id

        for child_nodes in node.children.values():
            path_tuple = (create_node(node, create=False),
                          create_node(child_nodes, create=False))

            self.construct_tree(dot, child_nodes, show_only_used_edges)
            if show_only_used_edges and "none" in path_tuple[1].lower() and "none" in path_tuple[0].lower():
                continue
            generated_current_root = create_node(node, create=True)
            dot.edge(generated_current_root, create_node(child_nodes))

    def draw(self, show_only_used_edges=False, view_only=False):
        dot = Digraph(comment='The search tree', format='png')
        self.construct_tree(dot, self.originale_root, show_only_used_edges)
        file_name = 'search-tree_{time}'.format(time=time.time())
        if view_only:
            dot.view(tempfile.mktemp('.gv'))
        else:
            dot.render(file_name)
            return file_name + '.png'
