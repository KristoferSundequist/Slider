from policy import *
from util import *
from logger import *
from typing import *

import numpy as np
import torch

# batch == [([initial_states], [(value, reward, action, search_policy, isNotDone)])]


def prepare_targets(targets: [[(float, float, int, [float], bool)]], num_unroll_steps: int, action_space_size: int):
    one_hot_actions = [torch.stack([onehot(action_space_size, trajectory[i][2])
                                    for trajectory in targets]) for i in range(num_unroll_steps)]

    search_policies = [torch.tensor([trajectory[i][3]
                                     for trajectory in targets]).float() for i in range(num_unroll_steps)]

    # value_targets = [torch.stack([discretize(trajectory[i][0])
    #                              for trajectory in targets]).float() for i in range(num_unroll_steps)]
    value_targets = [torch.tensor([[trajectory[i][0]]
                                   for trajectory in targets]).float() for i in range(num_unroll_steps)]

    # observed_rewards = [torch.stack([discretize(trajectory[i][1])
    #                                 for trajectory in targets]).float() for i in range(num_unroll_steps)]
    observed_rewards = [torch.tensor([[trajectory[i][1]]
                                      for trajectory in targets]).float() for i in range(num_unroll_steps)]

    isNotDone = [torch.tensor([[trajectory[i][4]]
                               for trajectory in targets]).float() for i in range(num_unroll_steps)]

    return one_hot_actions, search_policies, value_targets, observed_rewards, isNotDone


def test_prepare_targets():
    num_unroll_steps = 5
    targets = [[(e*100 + s, s, np.random.randint(4), [s, e, 0.5, 0], e == 3)
                for s in range(num_unroll_steps)] for e in range(10)]

    one_hot_actions, search_policies, value_targets, observed_rewards, isNotDone = prepare_targets(targets, num_unroll_steps, 4)

    assert len(one_hot_actions) == num_unroll_steps
    assert one_hot_actions[0].shape == torch.Size([10, 4])

    assert len(search_policies) == num_unroll_steps
    assert search_policies[0].shape == torch.Size([10, 4])

    assert all([torch.equal(search_policies[i][:, 0], torch.ones(10)*i) for i in range(num_unroll_steps)])

    assert value_targets[0].shape == torch.Size([10, 1])
    #assert value_targets[0].shape == torch.Size([10, 41])
    #assert all([value_targets[i][j].sum() == 1 for i in range(num_unroll_steps) for j in range(10)])
    #assert value_targets[0][0][20] == 1

    assert observed_rewards[0].shape == torch.Size([10, 1])
    #assert observed_rewards[0].shape == torch.Size([10, 41])
    # assert all([torch.equal(observed_rewards[i][:, 0], torch.ones(10, dtype=torch.float)*i)
    #            ror i in range(num_unroll_steps)])
    #assert observed_rewards[0][0].sum() == 1
    #assert all([observed_rewards[i][j].sum() == 1 for i in range(num_unroll_steps) for j in range(10)])
    #assert observed_rewards[0][0][20] == 1
    #assert observed_rewards[1][0][30] == 1

    assert isNotDone[0].shape == torch.Size([10, 1])
    assert sum([x.sum() for x in isNotDone]) == 5

def hook_fn(grad):
    return grad/2

def train_on_batch(
    batch: [([np.ndarray], [(float, float, int, [float], bool)])],
    representation: Representation,
    dynamics: Dynamics,
    prediction: Prediction,
    representation_optimizer,
    dynamics_optimizer,
    prediction_optimizer,
    projector,
    projector_optimizer,
    simpredictor,
    simpredictor_optimizer,
    action_space_size: int,
    num_unroll_steps: int,
    num_initial_states: int,
    logger: Logger
):
    scalar_loss = nn.L1Loss()
    cosine_sim = nn.CosineSimilarity()
    policy_loss_scale = 1

    targets = [e[1] for e in batch]
    one_hot_actions, search_policies, value_targets, observed_rewards, isNotDone = prepare_targets(targets, num_unroll_steps, action_space_size)

    initial_states = [e[0][:num_initial_states] for e in batch]
    inner_states = representation.forward(representation.prepare_states(initial_states))
    policy, value = prediction.forward(inner_states)

    assert policy.shape == search_policies[0].shape
    policy_loss = categorical_cross_entropy(policy, search_policies[0])*policy_loss_scale

    assert value.shape == value_targets[0].shape
    #value_loss = categorical_cross_entropy(value,value_targets[0])
    value_loss = scalar_loss(value,value_targets[0])
    loss = policy_loss + value_loss

    pl = 0
    vl = 0
    rl = 0
    sl = 0
    for i in range(num_unroll_steps-1):
        reward, inner_states = dynamics.forward(inner_states, one_hot_actions[i])

        # similarity loss
        with torch.no_grad():
            real_states = [e[0][(i+1):(i+num_initial_states+1)] for e in batch]
            repr_state = representation.forward(representation.prepare_states(real_states))
            repr_projector_state = projector.forward(repr_state).detach()
        sim_inner_projector_state = projector.forward(inner_states)
        sim_predictor_state = simpredictor.forward(sim_inner_projector_state)
        sim_loss = -cosine_sim(sim_predictor_state, repr_projector_state).mean()

        assert reward.shape == observed_rewards[i].shape
        #reward_loss = categorical_cross_entropy(reward, observed_rewards[i])
        reward_loss = scalar_loss(reward, observed_rewards[i])

        inner_states.register_hook(hook_fn)
        policy, value = prediction.forward(inner_states)

        assert policy.shape == search_policies[i+1].shape
        policy_loss = categorical_cross_entropy(policy, search_policies[i+1])*policy_loss_scale

        assert value.shape == value_targets[i+1].shape
        #value_loss = categorical_cross_entropy(value, value_targets[i+1])
        value_loss = scalar_loss(value, value_targets[i+1])

        step_loss = (policy_loss + value_loss + reward_loss + sim_loss)/num_unroll_steps

        pl += policy_loss.item()
        vl += value_loss.item()
        rl += reward_loss.item()
        sl += sim_loss.item()

        loss += step_loss
        # TODO: dont add to loss if isNotDone

    logger.add_head_losses(loss.item(), pl/num_unroll_steps, vl/num_unroll_steps, rl/num_unroll_steps, sl/num_unroll_steps)
    representation.zero_grad()
    dynamics.zero_grad()
    prediction.zero_grad()
    projector.zero_grad()
    simpredictor.zero_grad()

    loss.backward()

    max_grad_norm = 0.5
    torch.nn.utils.clip_grad_norm_(representation.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(prediction.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(simpredictor.parameters(), max_grad_norm)

    representation_optimizer.step()
    dynamics_optimizer.step()
    prediction_optimizer.step()
    projector_optimizer.step()
    simpredictor_optimizer.step()

'''

TEST cosineloss

'''


def test_cosineloss():
    cosine_loss = nn.CosineSimilarity()
    first = torch.Tensor([[1,2,3],[4,5,6]])
    second = torch.tensor([[1,2,3],[4,5,2]])
    sim_loss = -cosine_loss(first, second)
    assert -cosine_loss(first, first).mean() < -cosine_loss(first, second).mean()
