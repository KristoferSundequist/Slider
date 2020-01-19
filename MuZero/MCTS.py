import numpy as np
import torch
from policy import *
from Node import *
from util import *


def forward_simulation(
    root: Node,
    dynamics: Dynamics,
    prediction: Prediction,
    action_space_size: int,
    normalizer: Normalizer
):

    path = []
    actions = []

    current_node = root

    while True:
        path.append(current_node)
        action = current_node.get_action(normalizer)
        actions.append(action)
        child_node = current_node.edges[action]
        if child_node == None:
            with torch.no_grad():
                reward, new_state = dynamics.forward(
                    current_node.inner_state, onehot(action_space_size, action).unsqueeze(0))
                policy, value = prediction.forward(new_state)
                new_node = Node(new_state, action_space_size, policy.numpy().squeeze())
                #current_node.expand(action, expected_value(reward).item(), new_node)
                current_node.expand(action, reward.item(), new_node)
                # path.append(new_node)
                #return (path, actions, expected_value(value).item())
                return (path, actions, value.item())
        else:
            current_node = child_node


def backwards(path: [Node], actions: [int], value: float, discount: float, normalizer: Normalizer):
    rev_path = reversed(path)
    rev_actions = reversed(actions)

    for (node, action) in zip(rev_path, rev_actions):
        node.update(value, action)
        normalizer.update(node.mean_values[action])
        value = node.rewards[action] + discount * value


def get_noise(policy: np.ndarray, alpha: float = 0.25, frac:float = 0.25) -> np.ndarray:
    actions = list(range(len(policy)))
    noise = np.random.dirichlet([alpha] * len(actions))
    return np.array([policy[a] * (1 - frac) + n * frac for (a, n) in zip(actions, noise)])

def MCTS(
        initial_states: [np.ndarray],
        representation: Representation,
        dynamics: Dynamics,
        prediction: Prediction,
        action_space_size: int,
        num_simulations: int,
        discount: float
    ):

    normalizer = Normalizer(0, 10)

    with torch.no_grad():
        inner = representation.forward(representation.prepare_states(initial_states))
        policy, _ = prediction.forward(inner)

        root = Node(inner, action_space_size, get_noise(policy.numpy().squeeze(), 0.25, 0.25))
        #root.mean_values = [1.7,0,0,0]
        #root.visit_counts = [1,0,0,0]
        #root = Node(inner, action_space_size, np.array([0.1,0.1,0.1,0.7]))

        for i in range(num_simulations):
            (path, actions, value) = forward_simulation(root, dynamics,
                                                        prediction, action_space_size, normalizer)
            backwards(path, actions, value, discount, normalizer)

        return root

def sample_action(root: Node, temperature: float = 1) -> int:
    probs = root.get_search_policy(temperature)
    return np.random.choice(len(probs), 1, p=probs).item()

def get_best_action(root: Node) -> int:
    return np.argmax(root.visit_counts)
    
    


# HELP FUNCS

def print_tree(node: Node, depth=0, max_depth = 10000):
    if depth > max_depth:
        return

    prefix = "-" * depth
    visit_counts = sum(node.visit_counts)
    print(prefix, "value:", node.search_value() if visit_counts > 0 else None )
    print(prefix, "search_policy", node.get_search_policy() if visit_counts > 0 else None)
    print(prefix, "prior_policy:", node.policy)
    print(prefix, "visit_counts:", node.visit_counts)
    print(prefix, "mean_values:", node.mean_values)
    print(prefix, "rewards:", node.rewards)
    print("-------------------------------CHILDREN---------------------------------")
    for a in range(node.action_space_size):
        if node.edges[a] is None:
            print(prefix, a, None)
        else:
            print("--------------------------------NODE-----------------------------------")
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
        action_count[sample_action(root, 1)] += 1
    
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
    assert True

'''
    TEST get_noise()
'''

def test_get_noise():
    p = np.array([0.25,0.25,0.25,0.25])
    new_p = get_noise(p)
    assert new_p.shape == p.shape