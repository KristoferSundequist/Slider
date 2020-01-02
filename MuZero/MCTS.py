import numpy as np
import torch
from policy import *
from Node import *
from util import *


def forward_simulation(
    root: Node,
    dynamics: Dynamics,
    prediction: Prediction,
    action_space_size: int
):

    path = []
    actions = []

    current_node = root

    while True:
        path.append(current_node)
        action = current_node.get_action()
        actions.append(action)
        child_node = current_node.edges[action]
        if child_node == None:
            reward, new_state = dynamics.forward(
                current_node.inner_state, onehot(action_space_size, action).unsqueeze(0))
            policy, value = prediction.forward(new_state)
            new_node = Node(new_state, action_space_size, policy.numpy().squeeze())
            current_node.expand(action, reward.item(), new_node)
            # path.append(new_node)
            return (path, actions, value.item())
        else:
            current_node = child_node


def backwards(path: [Node], actions: [int], value: float, discount: float):
    rev_path = reversed(path)
    rev_actions = reversed(actions)

    for (node, action) in zip(rev_path, rev_actions):
        node.update(value, action)
        value = node.rewards[action] + discount * value


def MCTS(
        initial_states: [np.ndarray],
        representation: Representation,
        dynamics: Dynamics,
        prediction: Prediction,
        action_space_size: int,
        num_simulations: int,
        discount: float
    ):

    with torch.no_grad():
        inner = representation.forward(representation.prepare_states(initial_states))
        policy, value = prediction.forward(inner)
        #root = Node(inner, action_space_size, policy.numpy().squeeze())
        root = Node(inner, action_space_size, np.array([0.1,0.1,0.1,0.7]))

        for i in range(num_simulations):
            (path, actions, value) = forward_simulation(root, dynamics,
                                                        prediction, action_space_size)
            backwards(path, actions, value, discount)

        return root

def sample_action(root: Node, temperature: float = 1) -> int:
    temperature = 1/temperature
    total_visits = sum(map(lambda v: v**temperature, root.visit_counts))
    probs = list(map(lambda k: (k**temperature)/total_visits, root.visit_counts))
    return np.random.choice(len(probs), 1, p=probs).item()

def get_best_action(root: Node) -> int:
    return np.argmax(root.visit_counts)
    
    


# HELP FUNCS

def print_tree(node: Node, depth=0, max_depth = 10000):
    if depth > max_depth:
        return

    prefix = "-" * depth
    print("--------------------------------NODE-----------------------------------")
    print(prefix, "value:", node.value())
    print(prefix, "search_policy", node.get_search_policy())
    print(prefix, "prior_policy:", node.policy)
    print("-------------------------------CHILDREN---------------------------------")
    for a in range(node.action_space_size):
        if node.edges[a] is None:
            print(prefix, a, None)
        else:
            print(prefix, a, "visit count", node.visit_counts[a], "mean_value", node.mean_values[a], "reward", node.rewards[a])
            print_tree(node.edges[a], depth+1, max_depth)

def count_children(node: Node) -> int:
    children = list(filter(lambda v: v != None, node.edges))
    return len(children) + sum([count_children(c) for c in children])


# TESTS

def test_mcts_forward():
    num_initial_states = 13
    state_space_size = 6
    inner_size = 20
    action_space_size = 4
    batch_size = 1

    r = Representation(num_initial_states, state_space_size, inner_size)
    d = Dynamics(inner_size, action_space_size)
    p = Prediction(inner_size, action_space_size)

    raw_states = [[np.random.rand(state_space_size)
                   for i in range(num_initial_states)]
                  for _ in range(batch_size)]

    root = MCTS(raw_states, r, d, p, action_space_size, 50, .99)

    assert count_children(root) == 50


'''

    GET ACTION TESTS

'''

def test_sample_action():
    root = Node(torch.rand(4), 4, np.random.rand(4))
    root.visit_counts = [40,10,30,20]

    action_count = [0,0,0,0]
    for _ in range(10000):
        action_count[sample_action(root)] += 1
    
    action_count_rounded_to_nearest_thousand = list(map(lambda v: round(v, -3), action_count))
    
    assert action_count_rounded_to_nearest_thousand == [4000,1000,3000,2000]

def test_sample_action_with_temperature():
    root = Node(torch.rand(4), 4, np.random.rand(4))
    root.visit_counts = [40,10,30,20]

    action_count = [0,0,0,0]
    for _ in range(10000):
        action_count[sample_action(root, 0.2)] += 1
    
    action_count_rounded_to_nearest_thousand = list(map(lambda v: round(v, -3), action_count))
    
    assert action_count_rounded_to_nearest_thousand == [8000,0,2000,0]

def test_get_best_action():
    root = Node(torch.rand(4), 4, np.random.rand(4))
    root.visit_counts = [10,40,30,20]

    assert get_best_action(root) == 1

'''

    PRINT TREE TESTS

'''

def test_print_tree_simple():
    root = Node(torch.rand(4), 4, np.random.rand(4))
    child = Node(torch.rand(4), 4, np.random.rand(4))
    root.expand(1, 4, child)
    root.visit_counts = [2,3,2,1]
    child.visit_counts = [2,3,2,1]
    print_tree(root)
    assert True


def test_print_tree_MCTS():
    num_initial_states = 13
    state_space_size = 6
    inner_size = 20
    action_space_size = 4
    batch_size = 1

    r = Representation(num_initial_states, state_space_size, inner_size)
    d = Dynamics(inner_size, action_space_size)
    p = Prediction(inner_size, action_space_size)

    raw_states = [[np.random.rand(state_space_size)
                   for i in range(num_initial_states)]
                  for _ in range(batch_size)]

    root = MCTS(raw_states, r, d, p, action_space_size, 301, 0.99)
    print_tree(root, 0, 0)
    assert False
