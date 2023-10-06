import slider

import time
import torch
from torch import nn
from graphics import *
from replayBuffer import ReplayBuffer
import random
import math
import torch.nn.functional as F
from sequenceBuffer import SequenceBuffer, Sequence
import globals
import numpy as np
from sequenceBuffer import *
from logger import *
from utils import calculate_value_targets_for_batch
from representationNetwork import RepresentationNetwork
from policyNetwork import PolicyNetwork
from reconstructionNetwork import ReconstructionNetwork
from rewardNetwork import RewardNetwork
from transitionNetwork import TransitionNetwork
from valueNetwork import ValueNetwork
from recurrentNetwork import RecurrentnNetwork
from torch.distributions import Categorical, OneHotCategorical

# game = slider.GameWithoutSpeed
# game = simple_slider.Game
game = slider.Game

representationNetwork = RepresentationNetwork(game.state_space_size).to(globals.device)
policyNetwork = PolicyNetwork(game.action_space_size).to(globals.device)
reconstructionNetwork = ReconstructionNetwork(game.state_space_size).to(globals.device)
rewardNetwork = RewardNetwork().to(globals.device)
transitionNetwork = TransitionNetwork().to(globals.device)
valueNetwork = ValueNetwork().to(globals.device)
targetValueNetwork = copy.deepcopy(valueNetwork)
recurrentNetwork = RecurrentnNetwork(game.action_space_size).to(globals.device)

replay_buffer = ReplayBuffer(globals.replay_buffer_size)


# export OMP_NUM_THREADS=1


logger = Logger()


def combine_states(stoch_states: torch.Tensor, recurrent_states: torch.Tensor) -> torch.Tensor:
    return torch.concat([stoch_states, recurrent_states], 1)


def live(iterations: int, should_improve: bool, should_render: bool, should_visualize_reconstruction: bool):
    global targetValueNetwork

    win: GraphWin | None = None
    if should_render:
        win = GraphWin("Real game", globals.width, globals.height)
        win.setBackground("lightskyblue")

    reconstruction_win: GraphWin | None = None
    transition_win: GraphWin | None = None
    if should_visualize_reconstruction:
        reconstruction_win = GraphWin("Reconstructed from representation", globals.width, globals.height)
        reconstruction_win.setBackground("lightskyblue")

        transition_win = GraphWin("Reconstructed from another transition", globals.width, globals.height)
        transition_win.setBackground("lightskyblue")

    sequenceBuffer = SequenceBuffer(globals.sequence_length)
    g = game()
    reconstruction_game = game()
    transition_game = game()
    total_episode_reward = 0
    start = time.time()

    state = g.get_state()
    stoch_hidden_state = representationNetwork.get_initial(1)
    recurrent_hidden_state = recurrentNetwork.get_initial(1)
    action = 0
    for i in range(iterations):
        value = torch.tensor([0])
        reward_belief = torch.tensor([0])
        with torch.no_grad():
            action_tensor = F.one_hot(torch.tensor([action]), game.action_space_size).float().to(globals.device)
            recurrent_hidden_state = recurrentNetwork.forward(action_tensor, stoch_hidden_state, recurrent_hidden_state)
            stoch_hidden_state = representationNetwork.forward(
                torch.tensor([state]).to(globals.device), recurrent_hidden_state
            )
            combined_hidden = combine_states(stoch_hidden_state, recurrent_hidden_state)
            action = Categorical(logits=policyNetwork.forward(combined_hidden)).sample().item()
            if should_render:
                reward_belief = rewardNetwork.forward(combined_hidden).squeeze()
                value = valueNetwork.forward(combined_hidden).squeeze()

        if should_render:
            g.render(round(value.item(), 2), round(reward_belief.item(), 2), win)

        if should_visualize_reconstruction:
            with torch.no_grad():
                reconstructed_state = reconstructionNetwork.forward(combined_hidden).squeeze().tolist()
                reconstruction_game.set_game_state(reconstructed_state)
                reconstruction_game.render(0, 0, reconstruction_win)

                trans_stoch_state = stoch_hidden_state
                trans_recurrent_state = recurrent_hidden_state
                for _ in range(globals.imagination_horizon):
                    trans_recurrent_state = recurrentNetwork.forward(
                        F.one_hot(torch.tensor([3]), game.action_space_size).float().to(globals.device),
                        trans_stoch_state,
                        trans_recurrent_state,
                    )
                    trans_stoch_state = transitionNetwork.forward(trans_recurrent_state)
                    combined_transition_hidden = combine_states(trans_stoch_state, trans_recurrent_state)
                    reconstructed_state_from_transition = (
                        reconstructionNetwork.forward(combined_transition_hidden).squeeze().tolist()
                    )
                    transition_value = valueNetwork.forward(combined_transition_hidden)
                    transition_reward = rewardNetwork.forward(combined_transition_hidden)
                    transition_game.set_game_state(reconstructed_state_from_transition)
                    transition_game.render(
                        round(transition_value.item(), 2), round(transition_reward.item(), 2), transition_win
                    )
                    time.sleep(0.01)

        reward, next_state = g.step(action)

        sequenceBuffer.push(state, action, reward)

        state = next_state

        if i > globals.sequence_length:
            replay_buffer.push(sequenceBuffer.get())

        if i != 0 and len(replay_buffer) > 1000 and should_improve and i % globals.update_frequency == 0:
            improve(replay_buffer.sample(globals.batch_size))
            if i % (globals.update_frequency * 200) == 0:
                targetValueNetwork = copy.deepcopy(valueNetwork)

        total_episode_reward += reward

    logger.add_reward(total_episode_reward)
    print(f"avg_running_reward: {logger.get_running_avg_reward()}, total_episode_reward: {total_episode_reward}")

    end = time.time()
    print("Elapsed time: ", end - start)
    if win is not None:
        win.close()

    if reconstruction_win is not None:
        reconstruction_win.close()

    if transition_win is not None:
        transition_win.close()


def improve(observed_sequences: List[Sequence]):
    total_reconstruction_loss: torch.Tensor = torch.tensor(0.0).to(globals.device)
    total_reward_loss: torch.Tensor = torch.tensor(0.0).to(globals.device)
    total_transition_loss: torch.Tensor = torch.tensor(0.0).to(globals.device)
    total_representation_loss: torch.Tensor = torch.tensor(0.0).to(globals.device)

    saved_recurrent_states = []
    saved_stoch_states = []

    stoch_hidden_states = representationNetwork.get_initial(len(observed_sequences))
    recurrent_hidden_states = recurrentNetwork.get_initial(len(observed_sequences))
    combined_states = combine_states(stoch_hidden_states, recurrent_hidden_states)

    for i in range(globals.sequence_length):
        # prepare observed stuff
        observed_states = (
            torch.tensor([observation.observations[i] for observation in observed_sequences]).float().to(globals.device)
        )
        onehot_taken_actions = (
            F.one_hot(
                torch.tensor([observation.actions[i] for observation in observed_sequences]), game.action_space_size
            )
            .float()
            .to(globals.device)
        )
        observed_rewards = (
            torch.tensor([[observation.rewards[i]] for observation in observed_sequences]).float().to(globals.device)
        )

        # get next hidden states
        next_recurrent_states = recurrentNetwork.forward(
            onehot_taken_actions, stoch_hidden_states, recurrent_hidden_states
        )
        next_stoch_states = representationNetwork.forward(observed_states, next_recurrent_states)

        # calculate losses
        if i > 0:
            # reconstruction loss
            reconstructed_hidden_states = reconstructionNetwork.forward(combined_states)
            assert reconstructed_hidden_states.size() == observed_states.size()
            reconstruction_loss = 0.5 * (reconstructed_hidden_states - observed_states).pow(2).sum(1).mean()

            # reward loss
            predicted_rewards = rewardNetwork.forward(combined_states)
            assert predicted_rewards.size() == observed_rewards.size()
            reward_loss = 0.5 * (predicted_rewards - observed_rewards).pow(2).mean()

            # transition loss
            next_transition_stoch_states = transitionNetwork.forward(next_recurrent_states)
            assert next_transition_stoch_states.size() == next_stoch_states.size()
            transition_loss = 0.5 * (next_transition_stoch_states - next_stoch_states.detach()).pow(2).sum(1).mean()
            representation_loss = 0.5 * (next_transition_stoch_states.detach() - next_stoch_states).pow(2).sum(1).mean()

            total_reconstruction_loss += reconstruction_loss
            total_reward_loss += reward_loss
            total_transition_loss += 0.5 * transition_loss
            total_representation_loss += 0.1 * representation_loss

        recurrent_hidden_states = next_recurrent_states
        stoch_hidden_states = next_stoch_states + torch.normal(torch.zeros_like(next_stoch_states), 0.01)
        combined_states = combine_states(stoch_hidden_states, recurrent_hidden_states)

        if i % 2 == 0:
            saved_recurrent_states.append(recurrent_hidden_states.detach())
            saved_stoch_states.append(stoch_hidden_states.detach())

    logger.add_reconstuction_loss(total_reconstruction_loss.item())
    logger.add_reward_loss(total_reward_loss.item())
    logger.add_transition_loss(total_transition_loss.item())
    logger.add_representation_loss(total_representation_loss.item())

    representationNetwork.opt.zero_grad()
    reconstructionNetwork.opt.zero_grad()
    rewardNetwork.opt.zero_grad()
    transitionNetwork.opt.zero_grad()
    recurrentNetwork.opt.zero_grad()

    total_loss = total_reconstruction_loss + total_reward_loss + total_transition_loss + total_representation_loss
    total_loss.backward()

    max_norm_clip = 100
    nn.utils.clip_grad.clip_grad_value_(representationNetwork.parameters(), max_norm_clip)
    nn.utils.clip_grad.clip_grad_value_(reconstructionNetwork.parameters(), max_norm_clip)
    nn.utils.clip_grad.clip_grad_value_(rewardNetwork.parameters(), max_norm_clip)
    nn.utils.clip_grad.clip_grad_value_(transitionNetwork.parameters(), max_norm_clip)
    nn.utils.clip_grad.clip_grad_value_(recurrentNetwork.parameters(), max_norm_clip)

    representationNetwork.opt.step()
    reconstructionNetwork.opt.step()
    rewardNetwork.opt.step()
    transitionNetwork.opt.step()
    recurrentNetwork.opt.step()

    # BEHAVIOIR LEARNING

    saved_recurrent_states_tensor = torch.concat(saved_recurrent_states, 0)
    saved_stoch_states_tensor = torch.concat(saved_stoch_states, 0)
    improve_behaviour(saved_recurrent_states_tensor, saved_stoch_states_tensor)


def improve_behaviour(recurrent_states: torch.Tensor, stoch_states: torch.Tensor):
    # Imagine forward and store actions and values
    recurrent_states = recurrent_states.detach()
    stoch_states = stoch_states.detach()
    combined_states = combine_states(stoch_states, recurrent_states)

    all_one_hot_actions = []
    all_action_logits = []
    all_values = []
    all_lagged_values = []
    all_rewards: List[List[float]] = []

    for i in range(globals.imagination_horizon):
        values = valueNetwork.forward(combined_states).squeeze()
        policy_dist = OneHotCategorical(logits=policyNetwork.forward(combined_states))
        policy_logits = policy_dist.logits
        one_hot_actions = policy_dist.sample().float()
        with torch.no_grad():
            lagged_value_targets = targetValueNetwork.forward(combined_states).squeeze()
            rewards = rewardNetwork.forward(combined_states).squeeze().tolist()
            recurrent_states = recurrentNetwork.forward(one_hot_actions, stoch_states, recurrent_states)
            new_stoch_states = transitionNetwork.forward(recurrent_states)
            stoch_states = new_stoch_states + torch.normal(torch.zeros_like(new_stoch_states), 0.01)
            combined_states = combine_states(stoch_states, recurrent_states)

        all_values.append(values)
        all_lagged_values.append(lagged_value_targets)
        all_action_logits.append(policy_logits)
        all_one_hot_actions.append(one_hot_actions)
        all_rewards.append(rewards)

    tensor_values = torch.stack(all_values).T.to(globals.device)
    tensor_lagged_values = torch.stack(all_lagged_values).T.to(globals.device)
    tensor_rewards = torch.tensor(all_rewards).T.to(globals.device)

    # Calculate returns
    tensor_value_targets = calculate_value_targets_for_batch(tensor_rewards, tensor_lagged_values)
    assert tensor_value_targets.size()[1] == globals.imagination_horizon
    # assert tensor_value_targets.size() == (globals.sequence_length * globals.batch_size, globals.imagination_horizon)

    # Calculate value losses
    assert tensor_value_targets.size() == tensor_values.size()
    value_loss1 = 0.5 * (tensor_value_targets - tensor_values).pow(2)
    clipped = tensor_lagged_values + torch.clamp(tensor_values - tensor_lagged_values, -0.3, 0.3)
    value_loss2 = 0.5 * (clipped - tensor_value_targets).pow(2)
    value_loss = torch.max(value_loss1, value_loss2).mean()

    # Calculate policy losses
    advantages = (tensor_value_targets - tensor_values).detach()
    all_one_hot_actions_tensor = torch.stack(all_one_hot_actions).permute((1, 0, 2))
    all_actions_logits_tensor = torch.stack(all_action_logits).permute((1, 0, 2))
    assert all_one_hot_actions_tensor.size() == all_actions_logits_tensor.size()
    taken_action_logits = (all_one_hot_actions_tensor * all_actions_logits_tensor).sum(2)
    assert taken_action_logits.size() == advantages.size()
    policy_loss = torch.neg(taken_action_logits * advantages).mean()

    # entropy loss
    entropy_loss = -OneHotCategorical(logits=all_actions_logits_tensor).entropy().mean()

    logger.add_value_loss(value_loss.item())
    logger.add_policy_loss(policy_loss.item())
    logger.add_entropy_loss(entropy_loss.item())

    valueNetwork.opt.zero_grad()
    policyNetwork.opt.zero_grad()

    value_loss.backward()
    (policy_loss + globals.entropy_coeff * entropy_loss).backward()

    max_norm_clip = 100
    nn.utils.clip_grad.clip_grad_value_(valueNetwork.parameters(), max_norm_clip)
    nn.utils.clip_grad.clip_grad_value_(policyNetwork.parameters(), max_norm_clip)

    policyNetwork.opt.step()
    valueNetwork.opt.step()


def live_loop(lives, iterations):
    for i in range(lives):
        print(f"Iteration: {i} of {lives}")
        live(iterations, True, False, False)


def init(lives, iterations):
    for i in range(lives):
        print(f"Init: {i} of {lives}")
        live(iterations, False, False, False)
