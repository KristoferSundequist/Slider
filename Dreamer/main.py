import slider

import simple_slider
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
from utils import init_hidden, calculate_value_targets_for_batch
from representationNetwork import RepresentationNetwork
from policyNetwork import PolicyNetwork
from reconstructionNetwork import ReconstructionNetwork
from rewardNetwork import RewardNetwork
from transitionNetwork import TransitionNetwork
from valueNetwork import ValueNetwork
from torch.distributions import Categorical, OneHotCategorical

game = slider.Game
# game = simple_slider.Game

representationNetwork = RepresentationNetwork(game.state_space_size, game.action_space_size).to(globals.device)
policyNetwork = PolicyNetwork(game.action_space_size).to(globals.device)
reconstructionNetwork = ReconstructionNetwork(game.state_space_size).to(globals.device)
rewardNetwork = RewardNetwork().to(globals.device)
transitionNetwork = TransitionNetwork(game.action_space_size).to(globals.device)
valueNetwork = ValueNetwork().to(globals.device)
targetValueNetwork = copy.deepcopy(valueNetwork)

replay_buffer = ReplayBuffer(globals.replay_buffer_size)


# export OMP_NUM_THREADS=1


logger = Logger()


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
    hidden_state = init_hidden(1)
    action = 0
    for i in range(iterations):
        value = torch.tensor([0])
        reward_belief = torch.tensor([0])
        with torch.no_grad():
            hidden_state = representationNetwork.forward(
                torch.tensor([state]).to(globals.device),
                F.one_hot(torch.tensor([action]), game.action_space_size).float().to(globals.device),
                hidden_state,
            )
            action = Categorical(logits=policyNetwork.forward(hidden_state)).sample().item()
            if should_render:
                reward_belief = rewardNetwork.forward(hidden_state).squeeze()
                value = valueNetwork.forward(hidden_state).squeeze()

        if should_render:
            g.render(round(value.item(), 2), round(reward_belief.item(), 2), win)

        if should_visualize_reconstruction:
            with torch.no_grad():
                reconstructed_state = reconstructionNetwork.forward(hidden_state).squeeze().tolist()
                reconstruction_game.set_game_state(reconstructed_state)
                reconstruction_game.render(0, 0, reconstruction_win)

                transitioned_hidden = hidden_state
                for _ in range(globals.imagination_horizon):
                    transitioned_hidden = transitionNetwork.forward(
                        F.one_hot(torch.tensor([3]), game.action_space_size).float().to(globals.device),
                        transitioned_hidden,
                    )
                    reconstructed_state_from_transition = (
                        reconstructionNetwork.forward(transitioned_hidden).squeeze().tolist()
                    )
                    transition_value = valueNetwork.forward(transitioned_hidden)
                    transition_reward = rewardNetwork.forward(transitioned_hidden)
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

        if len(replay_buffer) > 1000 and should_improve and i % globals.update_frequency == 0:
            improve(replay_buffer.sample(globals.batch_size))
            if i % (globals.update_frequency * 100) == 0:
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

    saved_hidden_states = []

    hidden_states = init_hidden(len(observed_sequences)).detach()
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
        next_hidden_states = representationNetwork.forward(observed_states, onehot_taken_actions, hidden_states)

        # calculate losses
        if i > 0:
            # reconstruction loss
            reconstructed_hidden_states = reconstructionNetwork.forward(hidden_states)
            assert reconstructed_hidden_states.size() == observed_states.size()
            reconstruction_loss = 0.5 * (reconstructed_hidden_states - observed_states).pow(2).mean()

            # reward loss
            predicted_rewards = rewardNetwork.forward(hidden_states)
            assert predicted_rewards.size() == observed_rewards.size()
            reward_loss = 0.5 * (predicted_rewards - observed_rewards).pow(2).mean()

            # transition loss
            transition_hidden_states = transitionNetwork.forward(onehot_taken_actions, hidden_states)
            assert transition_hidden_states.size() == next_hidden_states.size()
            transition_loss = (
                globals.kl_balancing_alpha
                * (transition_hidden_states - next_hidden_states.detach()).pow(2).mean()
                * 0.5
                + (1 - globals.kl_balancing_alpha)
                * (transition_hidden_states.detach() - next_hidden_states).pow(2).mean()
                * 0.5
            )

            total_reconstruction_loss += reconstruction_loss
            total_reward_loss += reward_loss
            total_transition_loss += transition_loss

        hidden_states = next_hidden_states
        if i % 5 == 0:
            saved_hidden_states.append(hidden_states.detach())

    logger.add_reconstuction_loss(total_reconstruction_loss.item())
    logger.add_reward_loss(total_reward_loss.item())
    logger.add_transition_loss(total_transition_loss.item())

    representationNetwork.opt.zero_grad()
    reconstructionNetwork.opt.zero_grad()
    rewardNetwork.opt.zero_grad()
    transitionNetwork.opt.zero_grad()

    total_loss = total_reconstruction_loss + total_reward_loss + 0.5 * total_transition_loss
    total_loss.backward()

    max_norm_clip = 0.5
    nn.utils.clip_grad.clip_grad_norm_(representationNetwork.parameters(), max_norm_clip)
    nn.utils.clip_grad.clip_grad_norm_(reconstructionNetwork.parameters(), max_norm_clip)
    nn.utils.clip_grad.clip_grad_norm_(rewardNetwork.parameters(), max_norm_clip)
    nn.utils.clip_grad.clip_grad_norm_(transitionNetwork.parameters(), max_norm_clip)

    representationNetwork.opt.step()
    reconstructionNetwork.opt.step()
    rewardNetwork.opt.step()
    transitionNetwork.opt.step()

    # BEHAVIOIR LEARNING

    saved_hidden_states_tensor = torch.concat(saved_hidden_states, 0)
    improve_behaviour(saved_hidden_states_tensor)


def improve_behaviour(hidden_states: torch.Tensor):
    # Imagine forward and store actions and values
    hidden_states = hidden_states.detach()
    all_one_hot_actions = []
    all_action_probs = []
    all_values = []
    all_lagged_value_targets = []
    all_rewards: List[List[float]] = []

    for i in range(globals.imagination_horizon):
        values = valueNetwork.forward(hidden_states).squeeze()
        policy_dist = OneHotCategorical(logits=policyNetwork.forward(hidden_states))
        one_hot_actions = policy_dist.sample().float()
        with torch.no_grad():
            lagged_value_targets = targetValueNetwork.forward(hidden_states).squeeze()
            rewards = rewardNetwork.forward(hidden_states).squeeze().tolist()
            new_hidden_states = transitionNetwork.forward(one_hot_actions, hidden_states)
            hidden_states = new_hidden_states

        all_values.append(values)
        all_lagged_value_targets.append(lagged_value_targets)
        all_action_probs.append(policy_dist.probs)
        all_one_hot_actions.append(one_hot_actions)
        all_rewards.append(rewards)

    tensor_values = torch.stack(all_values).T.to(globals.device)
    tensor_lagged_value_targets = torch.stack(all_lagged_value_targets).T.to(globals.device)
    tensor_rewards = torch.tensor(all_rewards).T.to(globals.device)

    # Calculate returns
    tensor_value_targets = calculate_value_targets_for_batch(
        tensor_rewards, tensor_lagged_value_targets, globals.discount_factor, globals.keep_value_ratio
    )
    assert tensor_value_targets.size()[1] == globals.imagination_horizon
    # assert tensor_value_targets.size() == (globals.sequence_length * globals.batch_size, globals.imagination_horizon)

    # Calculate losses
    assert tensor_value_targets.size() == tensor_values.size()
    value_loss = 0.5 * (tensor_value_targets - tensor_values).pow(2).mean()

    advantages = (tensor_value_targets - tensor_values).detach()
    all_one_hot_actions_tensor = torch.stack(all_one_hot_actions).permute((1, 0, 2))
    all_actions_probs_tensor = torch.stack(all_action_probs).permute((1, 0, 2))
    assert all_one_hot_actions_tensor.size() == all_actions_probs_tensor.size()
    taken_action_probs = (all_one_hot_actions_tensor * all_actions_probs_tensor).sum(2)
    assert taken_action_probs.size() == advantages.size()
    policy_loss = torch.neg(torch.log(taken_action_probs) * advantages).mean()

    entropy = -(all_actions_probs_tensor * torch.log(all_actions_probs_tensor + 1e-08)).sum(2)
    entropy_loss = -entropy.mean()

    logger.add_value_loss(value_loss.item())
    logger.add_policy_loss(policy_loss.item())
    logger.add_entropy_loss(entropy_loss.item())

    valueNetwork.opt.zero_grad()
    policyNetwork.opt.zero_grad()

    value_loss.backward()
    (policy_loss + globals.entropy_coeff * entropy_loss).backward()

    nn.utils.clip_grad.clip_grad_norm_(valueNetwork.parameters(), 0.5)
    nn.utils.clip_grad.clip_grad_norm_(policyNetwork.parameters(), 0.5)

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
