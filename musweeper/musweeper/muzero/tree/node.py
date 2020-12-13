import random

import numpy as np
import torch


class min_max_node_tracker:
    def __init__(self):
        self.max = float('-inf')
        self.min = float('inf')

    def normalized(self, node_Q):
        """
        Normalize the value to [0, 1]

        Parameters
        ----------
        node_Q : float
                the node score form any node

        Returns
        -------
        float
                normalized score to [0, 1]
        """
        if self.min != self.max:
            # TODO : this shouldn't have to be called again. Find out why it is not called the first time.
            self.update(node_Q)
            return (node_Q - self.min) / (self.max - self.min)
        return node_Q

    def update(self, node_q):
        """
        Update the min-max tracker

        Parameters
        ----------
        node_q : float
                the node value
        """
        self.max = max(self.max, node_q)
        self.min = min(self.min, node_q)

    def __str__(self):
        return "min : {},  max : {}".format(self.min, self.max)

    def __repr__(self):
        return self.__str__()


class node:
    def __init__(self, parrent, node_id=None, hidden_state=None, prior=0):
        assert type(parrent) in [node, type(None)], type(parrent)
        assert hidden_state is None or torch.is_tensor(
            hidden_state), "{} {}".format(type(hidden_state), hidden_state)
        assert node_id is None or type(node_id) == int

        self.children = {}
        self.node_id = node_id
        self.parrent = parrent
        if self.parrent is not None and self.node_id not in self.parrent.children:
            self.parrent.children[self.node_id] = self

        self.min_max_node_tracker = min_max_node_tracker(
        ) if parrent is None else parrent.min_max_node_tracker

        self._value = 0
        self.value_sum = 0
        self.explored_count = 0
        self.wins_count = 0
        self.outcome = 0

        self.reward = 0
        self.policy = None
        self.prior = prior
        self.value_of_model = 0
        self.cumulative_discounted_reward = 0
        self.has_init = False

        self.hidden_state = hidden_state
        self.environment_state = None

        self.depth = 0 if parrent is None else (parrent.depth + 1)
        self.max_depth = self.depth

        self.available_children_paths = None
        self.score_metric = self.upper_confidence_boundary
        self.ucb_score_parts = [

        ]
        self.random_id = str(np.random.rand())

    def add_exploration_noise(self):
        """
        Add exploration noise as described in the paper in Appendix C 
        """
        dirichlet_alpha = 0.03
        root_exploration_fraction = 0.25
        child_actions = list(self.children.values())
        noise = np.random.dirichlet([dirichlet_alpha] * len(child_actions))
        for child_action, noise in zip(child_actions, noise):
            child_action.prior = child_action.prior * \
                (1 - root_exploration_fraction) + \
                noise * root_exploration_fraction

    def disable_illegal_actions(self, illegal_actions):
        if illegal_actions is None:
            return []
        # we just delete the illegal actions from the node
        for i in illegal_actions:
            action = i.item() if type(i) != int else i
            if action in self.children:
                del self.children[action]

    def search_value_exploration_exploration(self):
        """
        Nodes seelection algorithm
        As described in section "Exploration and exploitation" from https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

        Returns
        -------
        float
                the node score
        """
        parrent_explored = np.log2(self.parrent.explored_count) / \
            self.explored_count if self.parrent.explored_count != 1 and self.explored_count != 0 else 0
        child_explored = self.wins_count / \
            self.explored_count if self.explored_count > 0 else 0
        c = np.sqrt(2)

        return child_explored + c * np.sqrt(parrent_explored)

    def upper_confidence_boundary(self):
        """
        The upper confidene boundary
        as described in the appendix B of the paper.

        Returns
        -------
        float
                the upper confidence boundary
        """
        if self.parrent is None:
            return 0

        self.c1 = 1.25
        self.c2 = 19652

        self.q_s_a = self.q
        self.p_s_a = self.prior

        all_actions_sum = np.sum([
            i.explored_count for i in self.parrent.children.values()
        ])
        second_part_numerator_1 = np.sqrt(all_actions_sum)
        second_part_denominator_1 = (1 + self.explored_count)

        second_part_numerator_2 = (all_actions_sum + self.c2 + 1)
        second_part_denominator_2 = self.c2

        second_part = second_part_numerator_1 / second_part_denominator_1 * \
            (self.c1 + np.log(second_part_numerator_2 / second_part_denominator_2))

        value = self.q_s_a + self.p_s_a * second_part
        assert type(value) in [float, int, np.float64], "bad type {}, {}".format(
            type(value), value)
        self.ucb_score_parts = [
            self.q_s_a,
            self.p_s_a,
            all_actions_sum,
            second_part_numerator_1,
            second_part_denominator_1,
            second_part_numerator_2,
            second_part_denominator_2,
            second_part
        ]
        assert not np.isnan(value), "ucb score is nan {}".format(
            self.ucb_score_parts)
        return value

    @property
    def q(self):
        """
        Calculated the node value
        As described in appendix B

        Returns
        -------
        float
                node value score
        """
        reward = self.reward.item() if torch.is_tensor(self.reward) else self.reward
        node_value = self.node_value()
        value = self.min_max_node_tracker.normalized(
            node_value
        )
        assert type(reward) in [int, float]
        assert type(value) in [int, float]
        assert type(node_value) in [int, float]
        assert not np.isnan(reward), "reward is nan"
        assert not np.isnan(node_value), "node_value is nan"
        assert not np.isnan(value), "value is nan {}, {}".format(
            value, self.min_max_node_tracker)
        return reward + value

    @property
    def N(self):
        """
        Calculate the node visit count 

        Returns
        -------
        int
                node visit count
        """
        return self.parrent.explored_count + 1 if self.parrent else 0

    @property
    def value(self):
        """
                Return the value of the node

                Returns
                -------
                float
                        value of node (predicted by model)
                """
        return self.value_sum

    @value.setter
    def value(self, value):
        """
                Set the value

                Parameters
                ----------
                value : float
                        the value of the node
                """
        self.value_sum = value.item() if torch.is_tensor(value) else value
        self.min_max_node_tracker.update(self.node_value())

    def node_value(self):
        """
                The value of the node based on exploration

                Returns
                -------
                float
                        value divided by exploration count
                """
        if self.explored_count == 0:
            return 0
        return self.value_sum / self.explored_count

    def on_node_creation(self, hidden_state, reward, policy, value):
        """
        When a node is created this callback will be used

        Parameters
        ----------
        hidden_state : torch.tensor
                the hidden state from the model
        reward : float
                the reward from the environment
        """
        self.reward = reward
        self.hidden_state = hidden_state
        self.policy = policy
        self.value_of_model = value
        self.value = value
        self.has_init = True

        policy = policy[0] if len(policy.shape) > 1 else policy
        policy_sum = torch.sum(policy)
        self.prior = (torch.exp(policy[self.node_id]) / policy_sum).item()

    def get_a_children_node(self, children_count):
        """
        Returns a unexplored child node

        Parameters
        ----------
        children_count : int
                the count of available children

        Returns
        -------
        node
                the new child node
        """
        if self.available_children_paths is None:
            self.available_children_paths = list(
                filter(lambda x: x not in self.children, list(range(children_count))))
        if len(self.available_children_paths) == 0:
            return None
        picked_node = self.available_children_paths[random.randint(
            0, len(self.available_children_paths) - 1)]
        self.available_children_paths.remove(picked_node)
        return self.create_node(picked_node)

    def create_node(self, node_id):
        """
        Create a specific child node

        Parameters
        ----------
        node_id : int
                the action / node-id 

        Returns
        -------
        node
                the new node
        """
        self.children[node_id] = node(self, node_id=node_id)
        return self.children[node_id]

    def get_children_with_id(self, node_id):
        """
        Get node if it is a existing child node else none

        Parameters
        ----------
        node_id : int
                the node id

        Returns
        -------
        node
                the newly created node
        """
        return self.children.get(node_id, None)

    def create_children_if_not_exist(self, node_id):
        """
        Create node if it does not exist as child

        Parameters
        ----------
        node_id : int
                the node id

        Returns
        -------
        node
                the newly created node
        """
        node = self.get_children_with_id(node_id)
        if node is None:
            return self.create_node(node_id)
        return node

    def get_best_action(self):
        """
        Get the best available action based on children node score

        Returns
        -------
        int
                action
        """
        return max(self.children.items(), key=lambda x: x[1].search_value_exploration_exploration())[1].node_id

    def __str__(self):
        return "id : {}, value: {}, depth: {}".format(self.node_id, self.value_sum, self.depth)

    def __repr__(self):
        return self.__str__()
