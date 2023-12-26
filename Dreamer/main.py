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
from utils import calculate_value_targets_for_batch, get_average_gradient, combine_states
from representationNetwork import RepresentationNetwork
from policyNetwork import PolicyNetwork
from reconstructionNetwork import ReconstructionNetwork
from rewardNetwork import RewardNetwork
from transitionNetwork import TransitionNetwork
from valueNetwork import ValueHandler
from recurrentNetwork import RecurrentnNetwork
from torch.distributions import Categorical, OneHotCategorical, OneHotCategoricalStraightThrough

logger = Logger()
# game = slider.GameWithoutSpeed
# game = simple_slider.Game
game = slider.Game

representationNetwork = RepresentationNetwork(game.state_space_size).to(globals.device)
policyNetwork = PolicyNetwork(game.action_space_size, logger).to(globals.device)
reconstructionNetwork = ReconstructionNetwork(game.state_space_size).to(globals.device)
rewardNetwork = RewardNetwork().to(globals.device)
transitionNetwork = TransitionNetwork().to(globals.device)
valueHandler = ValueHandler(logger)
recurrentNetwork = RecurrentnNetwork(game.action_space_size).to(globals.device)

replay_buffer = ReplayBuffer(globals.replay_buffer_size)


# export OMP_NUM_THREADS=1


def train(iterations: int, num_parallell_games: int):
    games = [game() for _ in range(num_parallell_games)]
    states = [game.get_state() for game in games]
    stoch_hidden_state = representationNetwork.get_initial(num_parallell_games)
    recurrent_hidden_state = recurrentNetwork.get_initial(num_parallell_games)
    seq_buffers = [SequenceBuffer(globals.sequence_length) for _ in range(num_parallell_games)]
    actions = [0 for _ in range(num_parallell_games)]
    total_rewards = [0.0 for _ in range(num_parallell_games)]

    for i in range(iterations):
        with torch.no_grad():
            action_tensor = F.one_hot(torch.tensor(actions), game.action_space_size).float().to(globals.device)
            recurrent_hidden_state = recurrentNetwork.forward(action_tensor, stoch_hidden_state, recurrent_hidden_state)
            stoch_hidden_state = representationNetwork.forward(
                torch.tensor(states).to(globals.device), recurrent_hidden_state
            )
            combined_hidden = combine_states(stoch_hidden_state, recurrent_hidden_state)
            actions = Categorical(logits=policyNetwork.forward(combined_hidden)).sample().tolist()

        for ei in range(num_parallell_games):
            reward, next_state = games[ei].step(actions[ei])

            seq_buffers[ei].push(states[ei], actions[ei], reward)

            states[ei] = next_state
            if i > globals.sequence_length:
                replay_buffer.push(seq_buffers[ei].get())

            total_rewards[ei] += reward

        if i != 0 and len(replay_buffer) > 1000 and i % globals.update_frequency == 0:
            recurrent_rollouts, stoch_rollouts = improve_rssm(replay_buffer.sample(globals.batch_size))
            improve_behaviour(recurrent_rollouts, stoch_rollouts)

    for ei in range(num_parallell_games):
        logger.add("Reward", total_rewards[ei])


def live(iterations: int, should_improve: bool, should_render: bool, should_visualize_reconstruction: bool):
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
                value = valueHandler.forward(combined_hidden).squeeze()

        if should_render:
            g.render(round(value.item(), 4), round(reward_belief.item(), 4), win)

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
                    transition_value = valueHandler.forward(combined_transition_hidden)
                    transition_reward = rewardNetwork.forward(combined_transition_hidden)
                    transition_game.set_game_state(reconstructed_state_from_transition)
                    transition_game.render(
                        round(transition_value.item(), 4), round(transition_reward.item(), 4), transition_win
                    )
                    time.sleep(0.01)

        reward, next_state = g.step(action)

        sequenceBuffer.push(state, action, reward)

        state = next_state

        if i > globals.sequence_length:
            replay_buffer.push(sequenceBuffer.get())

        if i != 0 and len(replay_buffer) > 1000 and should_improve and i % globals.update_frequency == 0:
            recurrent_rollouts, stoch_rollouts = improve_rssm(replay_buffer.sample(globals.batch_size))
            improve_behaviour(recurrent_rollouts, stoch_rollouts)

        total_episode_reward += reward

    logger.add("Reward", total_episode_reward)
    print(f"avg_running_reward: {logger.get_running_avg('Reward')}, total_episode_reward: {total_episode_reward}")

    if win is not None:
        win.close()

    if reconstruction_win is not None:
        reconstruction_win.close()

    if transition_win is not None:
        transition_win.close()


def roll_rssm_forward(observed_sequences: List[Sequence]):
    saved_recurrent_states = []
    saved_stoch_states = []
    saved_observed_states = []
    saved_observed_rewards = []

    stoch_hidden_states = representationNetwork.get_initial(len(observed_sequences))
    recurrent_hidden_states = recurrentNetwork.get_initial(len(observed_sequences))

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

        # get next states
        recurrent_hidden_states = recurrentNetwork.forward(
            onehot_taken_actions, stoch_hidden_states, recurrent_hidden_states
        )
        stoch_hidden_states = representationNetwork.forward(observed_states, recurrent_hidden_states) + torch.normal(
            torch.zeros_like(stoch_hidden_states), 0.01
        )

        saved_recurrent_states.append(recurrent_hidden_states)
        saved_stoch_states.append(stoch_hidden_states)
        saved_observed_states.append(observed_states)
        saved_observed_rewards.append(observed_rewards)

    return (
        torch.concat(saved_recurrent_states, 0),
        torch.concat(saved_stoch_states, 0),
        torch.concat(saved_observed_states, 0),
        torch.concat(saved_observed_rewards, 0),
    )


def calculate_representation_losses(
    recurrent_rollouts: torch.Tensor,
    stoch_rollouts: torch.Tensor,
    observed_states: torch.Tensor,
    observed_rewards: torch.Tensor,
):
    combined_rollouts = combine_states(stoch_rollouts, recurrent_rollouts)

    # reconstruction loss
    reconstructed_hidden_states = reconstructionNetwork.forward(combined_rollouts)
    assert (
        reconstructed_hidden_states.size() == observed_states.size()
    ), f"{reconstructed_hidden_states.size()} == {observed_states.size()}"
    reconstruction_loss = 0.5 * (reconstructed_hidden_states - observed_states).pow(2).sum(1).mean()

    # reward loss
    predicted_rewards = rewardNetwork.forward(combined_rollouts)
    assert (
        predicted_rewards.size() == observed_rewards.size()
    ), f"{predicted_rewards.size()} == {observed_rewards.size()}"
    reward_loss = 0.5 * (predicted_rewards - observed_rewards).pow(2).mean()

    # transition loss
    transition_stoch_states = transitionNetwork.forward(recurrent_rollouts)
    assert (
        transition_stoch_states.size() == stoch_rollouts.size()
    ), f"{transition_stoch_states.size()} == {stoch_rollouts.size()}"
    transition_loss = 0.5 * (transition_stoch_states - stoch_rollouts.detach()).pow(2).sum(1).mean()
    representation_loss = 0.5 * (transition_stoch_states.detach() - stoch_rollouts).pow(2).sum(1).mean()

    scaled_reconstruction_loss = reconstruction_loss
    scaled_reward_loss = reward_loss
    scaled_transition_loss = 0.5 * transition_loss
    scaled_representation_loss = 0.1 * representation_loss

    logger.add("Reconstruction loss", scaled_reconstruction_loss.item())
    logger.add("Reward_loss", scaled_reward_loss.item())
    logger.add("Transition_loss", scaled_transition_loss.item())
    logger.add("Representation_loss", scaled_representation_loss.item())

    return scaled_reconstruction_loss + scaled_reward_loss + scaled_transition_loss + scaled_representation_loss


def update_repr(repr_loss: torch.Tensor):
    representationNetwork.opt.zero_grad()
    reconstructionNetwork.opt.zero_grad()
    rewardNetwork.opt.zero_grad()
    transitionNetwork.opt.zero_grad()
    recurrentNetwork.opt.zero_grad()

    repr_loss.backward()

    logger.add("Avg representation grad", get_average_gradient(representationNetwork))
    logger.add("Avg reconstruction grad", get_average_gradient(reconstructionNetwork))
    logger.add("Avg reward grad", get_average_gradient(rewardNetwork))
    logger.add("Avg transition grad", get_average_gradient(transitionNetwork))
    logger.add("Avg recurrent grad", get_average_gradient(recurrentNetwork))

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


def improve_rssm(observed_sequences: List[Sequence]):
    recurrent_rollouts, stoch_rollouts, observed_states, observed_rewards = roll_rssm_forward(observed_sequences)
    repr_loss = calculate_representation_losses(recurrent_rollouts, stoch_rollouts, observed_states, observed_rewards)
    update_repr(repr_loss)
    return recurrent_rollouts.detach(), stoch_rollouts.detach()


def roll_imagination_forward(recurrent_states: torch.Tensor, stoch_states: torch.Tensor):
    recurrent_states = recurrent_states.detach()
    stoch_states = stoch_states.detach()
    combined_states = combine_states(stoch_states, recurrent_states)

    saved_combined_states = []
    saved_one_hot_actions = []
    saved_policy_logits = []

    for i in range(globals.imagination_horizon):
        policy_dist = OneHotCategoricalStraightThrough(logits=policyNetwork.forward(combined_states))
        one_hot_actions = policy_dist.rsample()

        saved_combined_states.append(combined_states)
        saved_one_hot_actions.append(one_hot_actions)
        saved_policy_logits.append(policy_dist.logits)

        with torch.no_grad():
            recurrent_states = recurrentNetwork.forward(one_hot_actions, stoch_states, recurrent_states)
            stoch_states = transitionNetwork.forward(recurrent_states) + torch.normal(
                torch.zeros_like(stoch_states), 0.01
            )
            combined_states = combine_states(stoch_states, recurrent_states)

    # return [Batch, sequence, dim]
    return (
        torch.stack(saved_combined_states, 0).permute(1, 0, 2),
        torch.stack(saved_one_hot_actions, 0).permute(1, 0, 2),
        torch.stack(saved_policy_logits, 0).permute(1, 0, 2),
    )


def calculate_value_loss(
    values: torch.Tensor, lagged_values: torch.Tensor, value_targets: torch.Tensor
) -> torch.Tensor:
    assert value_targets.size() == values.size()
    value_loss1 = 0.5 * (value_targets - values).pow(2)
    clipped = lagged_values + torch.clamp(values - lagged_values, -0.3, 0.3)
    value_loss2 = 0.5 * (clipped - value_targets).pow(2)
    value_loss = torch.max(value_loss1, value_loss2).mean()
    logger.add("Value_loss", value_loss.item())
    return value_loss


def calculate_policy_loss(
    values: torch.Tensor, value_targets: torch.Tensor, one_hot_actions: torch.Tensor, policy_logits: torch.Tensor
):
    dist = OneHotCategorical(logits=policy_logits)

    # Policy loss
    advantages = (value_targets - values).detach()
    taken_action_logits = dist.log_prob(one_hot_actions)
    assert taken_action_logits.size() == advantages.size()
    policy_loss = -(taken_action_logits * advantages).mean()
    # assert one_hot_actions.size() == policy_logits.size()
    # taken_action_logits = (one_hot_actions * policy_logits).sum(2)
    # assert taken_action_logits.size() == advantages.size()
    # policy_loss = torch.neg(taken_action_logits * advantages).mean()

    # entropy loss
    entropy_loss = -dist.entropy().mean()
    scaled_entropy_loss = globals.entropy_coeff * entropy_loss

    logger.add("Policy_loss", policy_loss.item())
    logger.add("Entropy_loss", scaled_entropy_loss.item())

    return policy_loss + scaled_entropy_loss


def improve_behaviour(recurrent_states: torch.Tensor, stoch_states: torch.Tensor):
    combined_states, one_hot_actions, policy_logits = roll_imagination_forward(recurrent_states, stoch_states)
    values = valueHandler.forward(combined_states).squeeze()
    with torch.no_grad():
        lagged_values = valueHandler.forward_lagged(combined_states).squeeze()
        rewards = rewardNetwork.forward(combined_states).squeeze()

    # Calculate returns
    value_targets = calculate_value_targets_for_batch(rewards, lagged_values)
    assert value_targets.size()[1] == globals.imagination_horizon

    value_loss = calculate_value_loss(values, lagged_values, value_targets)
    policy_loss = calculate_policy_loss(values, value_targets, one_hot_actions, policy_logits)

    valueHandler.update(value_loss)
    policyNetwork.update(policy_loss)


def live_loop(lives, iterations):
    for i in range(lives):
        print(f"Iteration: {i} of {lives}")
        start = time.time()
        train(iterations, 6)
        end = time.time()
        print("Elapsed time: ", end - start)
        print(logger.get_avg_of_window("Reward", 6))
        print("------------------------")
        # live(iterations, True, False, False)


def train_rssm(n_batches: int):
    for i in range(n_batches):
        if i % (n_batches / 10) == 0:
            print(f"init training batch {i} of {n_batches}")
        improve_rssm(replay_buffer.sample(globals.batch_size))


def init(lives: int, iterations: int):
    for i in range(lives):
        live(iterations, False, False, False)

    train_rssm(round((lives * iterations) / globals.batch_size) * 2)
