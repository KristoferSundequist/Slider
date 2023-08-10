from policy import *
from util import *
from logger import *
from typing import *

import numpy as np
import torch
from episode import TrainingExample, TargetValues

# batch == [([initial_states], [(value, reward, action, search_policy, isNotDone)])]


def prepare_targets(trajectories: List[List[TargetValues]], num_unroll_steps: int, action_space_size: int):
    one_hot_actions = [torch.stack([onehot(action_space_size, targetValues[i].action)
                                    for targetValues in trajectories]) for i in range(num_unroll_steps)]

    search_policies = [torch.tensor([targetValues[i].search_policy
                                     for targetValues in trajectories]).float() for i in range(num_unroll_steps)]

    # value_targets = [torch.stack([discretize(trajectory[i][0])
    #                              for trajectory in targets]).float() for i in range(num_unroll_steps)]
    value_targets = [torch.tensor([[targetValues[i].value]
                                   for targetValues in trajectories]).float() for i in range(num_unroll_steps)]

    # observed_rewards = [torch.stack([discretize(trajectory[i][1])
    #                                 for trajectory in targets]).float() for i in range(num_unroll_steps)]
    observed_rewards = [torch.tensor([[targetValues[i].reward]
                                      for targetValues in trajectories]).float() for i in range(num_unroll_steps)]

    return one_hot_actions, search_policies, value_targets, observed_rewards


def test_prepare_targets():
    num_unroll_steps = 5
    targets = [[TargetValues(value=e*100 + s, reward=s, action=np.random.randint(4), search_policy=[s, e, 0.5, 0])
                for s in range(num_unroll_steps)] for e in range(10)]

    one_hot_actions, search_policies, value_targets, observed_rewards = prepare_targets(
        targets, num_unroll_steps, 4)

    assert len(one_hot_actions) == num_unroll_steps
    assert one_hot_actions[0].shape == torch.Size([10, 4])

    assert len(search_policies) == num_unroll_steps
    assert search_policies[0].shape == torch.Size([10, 4])

    assert all([torch.equal(search_policies[i][:, 0], torch.ones(10)*i)
               for i in range(num_unroll_steps)])

    assert value_targets[0].shape == torch.Size([10, 1])
    # assert value_targets[0].shape == torch.Size([10, 41])
    # assert all([value_targets[i][j].sum() == 1 for i in range(num_unroll_steps) for j in range(10)])
    # assert value_targets[0][0][20] == 1

    assert observed_rewards[0].shape == torch.Size([10, 1])
    # assert observed_rewards[0].shape == torch.Size([10, 41])
    # assert all([torch.equal(observed_rewards[i][:, 0], torch.ones(10, dtype=torch.float)*i)
    #            ror i in range(num_unroll_steps)])
    # assert observed_rewards[0][0].sum() == 1
    # assert all([observed_rewards[i][j].sum() == 1 for i in range(num_unroll_steps) for j in range(10)])
    # assert observed_rewards[0][0][20] == 1
    # assert observed_rewards[1][0][30] == 1


def hook_fn(grad):
    return grad/2


def train_on_batch(
    batch: List[TrainingExample],
    representation: Representation,
    dynamics: Dynamics,
    prediction: Prediction,
    projector,
    simpredictor,
    action_space_size: int,
    num_unroll_steps: int,
    num_initial_states: int,
    logger: Logger
):
    scalar_loss = nn.L1Loss()
    cosine_sim = nn.CosineSimilarity()
    policy_loss_scale = 0.1

    targets = [e.targetValues for e in batch]
    one_hot_actions, search_policies, value_targets, observed_rewards = prepare_targets(
        targets, num_unroll_steps, action_space_size)

    initial_states = [e.initialStates for e in batch]
    inner_states = representation.forward(
        representation.prepare_states(initial_states))
    policy, value = prediction.forward(inner_states)

    assert policy.shape == search_policies[0].shape
    policy_loss = categorical_cross_entropy(
        policy, search_policies[0])*policy_loss_scale

    assert value.shape == value_targets[0].shape
    # value_loss = categorical_cross_entropy(value,value_targets[0])
    value_loss = scalar_loss(value, value_targets[0])
    loss = policy_loss + value_loss

    pl = 0
    vl = 0
    rl = 0
    sl = 0
    for i in range(num_unroll_steps-1):
        reward, inner_states = dynamics.forward(
            inner_states, one_hot_actions[i])

        # similarity loss
        # with torch.no_grad():
        #    real_states = [e[0][(i+1):(i+num_initial_states+1)] for e in batch]
        #    repr_state = representation.forward(representation.prepare_states(real_states)).detach()
        # repr_projector_state = projector.forward(repr_state).detach()
        # sim_inner_projector_state = projector.forward(inner_states)
        # sim_predictor_state = simpredictor.forward(sim_inner_projector_state)
        # sim_loss = -cosine_sim(sim_predictor_state, repr_projector_state).mean()
        # sim_loss = -cosine_sim(inner_states, repr_state).mean()

        assert reward.shape == observed_rewards[i].shape
        # reward_loss = categorical_cross_entropy(reward, observed_rewards[i])
        reward_loss = scalar_loss(reward, observed_rewards[i])

        inner_states.register_hook(hook_fn)
        policy, value = prediction.forward(inner_states)

        assert policy.shape == search_policies[i+1].shape
        policy_loss = categorical_cross_entropy(
            policy, search_policies[i+1])*policy_loss_scale

        assert value.shape == value_targets[i+1].shape
        # value_loss = categorical_cross_entropy(value, value_targets[i+1])
        value_loss = scalar_loss(value, value_targets[i+1])

        # step_loss = (policy_loss + value_loss + reward_loss + sim_loss)/num_unroll_steps
        step_loss = (policy_loss + value_loss + reward_loss)/num_unroll_steps

        pl += policy_loss.item()
        vl += value_loss.item()
        rl += reward_loss.item()
        # sl += sim_loss.item()

        loss += step_loss
        # TODO: dont add to loss if isNotDone

    # logger.add_head_losses(loss.item(), pl/num_unroll_steps, vl/num_unroll_steps, rl/num_unroll_steps, sl/num_unroll_steps)
    logger.add_head_losses(loss.item(), pl/num_unroll_steps,
                           vl/num_unroll_steps, rl/num_unroll_steps, 0)

    batch_counter.increment()

    representation.opt.zero_grad()
    dynamics.opt.zero_grad()
    prediction.opt.zero_grad()
    # projector.opt.zero_grad()
    # simpredictor.opt.zero_grad()

    loss.backward()

    max_grad_norm = 1
    torch.nn.utils.clip_grad_norm_(representation.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_grad_norm)
    torch.nn.utils.clip_grad_norm_(prediction.parameters(), max_grad_norm)
    #torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
    #torch.nn.utils.clip_grad_norm_(simpredictor.parameters(), max_grad_norm)

    representation.opt.step()
    dynamics.opt.step()
    prediction.opt.step()
    # projector.step()
    # simpredictor.step()


'''

TEST cosineloss

'''


def test_cosineloss():
    cosine_loss = nn.CosineSimilarity()
    first = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    second = torch.tensor([[1, 2, 3], [4, 5, 2]])
    sim_loss = -cosine_loss(first, second)
    assert -cosine_loss(first, first).mean() < - \
        cosine_loss(first, second).mean()


def test_cosineloss_input_order_invariance():
    cosine_loss = nn.CosineSimilarity()
    first = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    second = torch.tensor([[1, 2, 3], [4, 5, 2]])
    sim_loss1 = -cosine_loss(first, second).mean()
    sim_loss2 = -cosine_loss(second, first).mean()
    assert sim_loss1 == sim_loss2
