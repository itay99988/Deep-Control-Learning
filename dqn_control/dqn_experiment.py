import sys
import math
import random
import numpy as np
from bisect import bisect
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from experiment import *
from dqn_control import DRQN


def average_soft(x):
    x /= np.sum(x)
    return x


def entropy(x):
    return -np.sum(x*np.log(x))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.rewards = deque([], maxlen=capacity)
        self.weight_dict = [1, 1, 1, 2, 2, 2, 2, 4, 4, 4]

    def push(self, sars_lst, reward):
        """Save a sars object"""
        self.memory.append(sars_lst)
        self.rewards.append(reward)

    def sample(self):
        """return one random element from the queue"""
        idx = np.random.choice(range(len(self)), size=1)[0]
        return random.sample(self.memory, BATCH_SIZE)

    def weighted_sample(self):
        distribution = self.get_rewards_dist()
        indices = np.random.choice(range(len(self)), size=BATCH_SIZE, p=distribution, replace=False)
        return [self.memory[i] for i in indices]

    def get_rewards_dist(self):
        # median = self.get_r_median()
        # weighted_r = [RE_MEM_WEIGHT if reward >= median else 1 for reward in self.rewards]
        deciles = np.percentile(self.rewards, range(10, 100, 10))
        weighted_r = [self.weight_dict[bisect(deciles, reward)] for reward in self.rewards]
        return average_soft(weighted_r)

    def get_r_softmax_dist(self):
        probs = np.array(F.softmax(torch.tensor(self.rewards, device=device) / 10, dim=-1))
        probs = probs / np.sum(probs)
        return probs

    def get_r_median(self):
        nums = sorted(self.rewards)
        middle1 = (len(nums) - 1) // 2
        middle2 = len(nums) // 2
        return (nums[int(middle1)] + nums[int(middle2)]) / 2

    def get_state_dim(self):
        return self.memory[0][0].state.shape[0]

    def __len__(self):
        return len(self.memory)


# experiment params
FIGURE_NUMBER = 7           if len(sys.argv) <= 1 else int(sys.argv[1])
TEST_REPETITIONS = 1        if len(sys.argv) <= 2 else int(sys.argv[2])
VERBOSITY = True            if len(sys.argv) <= 3 else int(sys.argv[3])
# hyperparameters
BATCH_SIZE = 2              if len(sys.argv) <= 4 else int(sys.argv[4])
GAMMA = 0.90                if len(sys.argv) <= 5 else float(sys.argv[5])
EPS_START = 0.9             if len(sys.argv) <= 6 else float(sys.argv[6])
EPS_END = 0.5               if len(sys.argv) <= 7 else float(sys.argv[7])
EPS_DECAY = 200             if len(sys.argv) <= 8 else int(sys.argv[8])
TARGET_UPDATE = 10          if len(sys.argv) <= 9 else int(sys.argv[9])
RNN_H_DIM = 10              if len(sys.argv) <= 10 else int(sys.argv[10])
HIDDEN_DIM = 7              if len(sys.argv) <= 11 else int(sys.argv[11])
LEARNING_RATE = 0.01        if len(sys.argv) <= 12 else float(sys.argv[12])
RE_MEM_WEIGHT = 4           if len(sys.argv) <= 13 else int(sys.argv[13])
LOSS = "Huber"              if len(sys.argv) <= 14 else str(sys.argv[14])
TEMPERATURE = 4000          if len(sys.argv) <= 15 else float(sys.argv[15])
REG_COEFF = 0.2

# training params
EPISODE_COUNT = 3000        if len(sys.argv) <= 16 else int(sys.argv[16])
EPISODE_LEN = 50            if len(sys.argv) <= 17 else int(sys.argv[17])
# evaluation params
EPISODE_EVAL_COUNT = 100    if len(sys.argv) <= 18 else int(sys.argv[18])
EPISODE_EVAL_LEN = 200      if len(sys.argv) <= 19 else int(sys.argv[19])

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# replay memory
Sars = namedtuple('Sars', ('state', 'action', 'next_state', 'reward'))


# select an action according to the policy network
def select_action(state, hidden_state, exploration=True):
    global steps_done

    available_actions_idx = [dist_sys.system.get_transition_idx(tr_name) for tr_name in dist_sys.system.available_transitions()]

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    #steps_done += 1

    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        state = state.view(1, 1, -1)
        action_values, hidden_state = policy_net(state, hidden_state)
        action_values = action_values.view(-1)
        filtered_action_values = action_values[available_actions_idx]

        max_idx = filtered_action_values.max(0)[1]

        if VERBOSITY:
            if steps_done % (EPISODE_LEN*10) == 1 and exploration:
                print("First step: {}".format(filtered_action_values))
            if steps_done % (EPISODE_LEN * 10) == 5 and exploration:
                print("Fifth step: {}".format(filtered_action_values))
            if steps_done % (EPISODE_LEN * 10) == 6 and exploration:
                print("Sixth step: {}".format(filtered_action_values))
            if steps_done % (EPISODE_LEN * 10) == 7 and exploration:
                print("Seventh step: {}".format(filtered_action_values))

    if not exploration or sample > eps_threshold:
        return torch.tensor([[available_actions_idx[max_idx]]], device=device), hidden_state
    else:
        return torch.tensor([[random.choice(available_actions_idx)]], device=device, dtype=torch.long), hidden_state


# select an action according to the policy network
def select_action_boltz(state, hidden_state):
    global steps_done
    global TEMPERATURE
    available_actions_idx = [dist_sys.system.get_transition_idx(tr_name) for tr_name in dist_sys.system.available_transitions()]

    sample = random.random()
    steps_done += 1
    T = max(TEMPERATURE / steps_done, 5)

    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        state = state.view(1, 1, -1)
        action_values, hidden_state = policy_net(state, hidden_state)
        action_values = action_values.view(-1)
        filtered_action_values = action_values[available_actions_idx]

        probs = np.array(F.softmax(filtered_action_values / T, dim=-1))
        probs = probs / np.sum(probs)

        if VERBOSITY:
            if steps_done % (EPISODE_LEN*10) == 1:
                print(probs)
                print("First step: {}".format(filtered_action_values))
            if steps_done % (EPISODE_LEN*10) == 5:
                print("Fifth step: {}".format(filtered_action_values))
            if steps_done % (EPISODE_LEN*10) == 6:
                print("Sixth step: {}".format(filtered_action_values))
            if steps_done % (EPISODE_LEN*10) == 7:
                print("Seventh step: {}".format(filtered_action_values))

    return torch.tensor([[np.random.choice(available_actions_idx, p=probs)]], device=device, dtype=torch.long), hidden_state


# optimization step
def optimize_model():
    global loss_lst

    if len(memory) < 10:
        return

    if LOSS == "Huber":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    policy_hidden_state = policy_net.init_hidden(batch_size=BATCH_SIZE)
    target_hidden_state = target_net.init_hidden(batch_size=BATCH_SIZE)

    # sars_lsts = memory.sample()
    sars_lsts = memory.weighted_sample()
    sars_joint = [sars_tuple for sublist in sars_lsts for sars_tuple in sublist]

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Sars objects
    # to Sars object of batch-arrays.

    episode_batch = Sars(*zip(*sars_joint))

    state_batch = torch.vstack(episode_batch.state).view(BATCH_SIZE, EPISODE_LEN, -1)
    action_batch = torch.cat(episode_batch.action).view(BATCH_SIZE, EPISODE_LEN, -1)
    reward_batch = torch.cat(episode_batch.reward).view(BATCH_SIZE, EPISODE_LEN)
    next_state_batch = torch.vstack(episode_batch.next_state).view(BATCH_SIZE, EPISODE_LEN, -1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_q_values = policy_net(state_batch, policy_hidden_state)[0]
    state_action_values = state_q_values.gather(2, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    next_state_q_values = target_net(next_state_batch, target_hidden_state)[0]
    next_state_values = next_state_q_values.max(2)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(2))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    loss_lst.append(loss.item())


# optimization step with a small correction
def optimize_model2():
    global loss_lst

    if len(memory) < 10:
        return

    if LOSS == "Huber":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    policy_hidden_state = policy_net.init_hidden(batch_size=BATCH_SIZE)
    target_hidden_state = target_net.init_hidden(batch_size=BATCH_SIZE)

    # sars_lsts = memory.sample()
    sars_lsts = memory.weighted_sample()
    sars_joint = [sars_tuple for sublist in sars_lsts for sars_tuple in sublist]

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Sars objects
    # to Sars object of batch-arrays.

    episode_batch = Sars(*zip(*sars_joint))
    # change the next state batch
    init_state = torch.tensor([0]*memory.get_state_dim(), dtype=torch.float)
    empty_st_lst = [init_state] * BATCH_SIZE
    updated_next_state_batch = sum([[empty_st_lst[i]] + list(episode_batch.next_state[EPISODE_LEN*i: EPISODE_LEN*i + EPISODE_LEN]) for i in range(BATCH_SIZE)], [])

    state_batch = torch.vstack(episode_batch.state).view(BATCH_SIZE, EPISODE_LEN, -1)
    action_batch = torch.cat(episode_batch.action).view(BATCH_SIZE, EPISODE_LEN, -1)
    reward_batch = torch.cat(episode_batch.reward).view(BATCH_SIZE, EPISODE_LEN)
    next_state_batch = torch.vstack(updated_next_state_batch).view(BATCH_SIZE, EPISODE_LEN + 1, -1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_q_values = policy_net(state_batch, policy_hidden_state)[0]
    state_action_values = state_q_values.gather(2, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    next_state_q_values = target_net(next_state_batch, target_hidden_state)[0]

    # We want everything but the first q values of each sequence
    next_state_q_values = next_state_q_values[:, 1:, :]

    next_state_values = next_state_q_values.max(2)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(2))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    loss_lst.append(loss.item())


# optimization step with double dqn
def optimize_model_ddqn():
    global loss_lst

    if len(memory) < 10:
        return

    if LOSS == "Huber":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    policy_hidden_state = policy_net.init_hidden(batch_size=BATCH_SIZE)
    target_hidden_state = target_net.init_hidden(batch_size=BATCH_SIZE)

    # sars_lsts = memory.sample()
    sars_lsts = memory.weighted_sample()
    sars_joint = [sars_tuple for sublist in sars_lsts for sars_tuple in sublist]

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Sars objects
    # to Sars object of batch-arrays.

    episode_batch = Sars(*zip(*sars_joint))
    # change the next state batch
    init_state = torch.tensor([0]*memory.get_state_dim(), dtype=torch.float)
    empty_st_lst = [init_state] * BATCH_SIZE
    updated_next_state_batch = sum([[empty_st_lst[i]] + list(episode_batch.next_state[EPISODE_LEN*i: EPISODE_LEN*i + EPISODE_LEN]) for i in range(BATCH_SIZE)], [])

    state_batch = torch.vstack(episode_batch.state).view(BATCH_SIZE, EPISODE_LEN, -1)
    action_batch = torch.cat(episode_batch.action).view(BATCH_SIZE, EPISODE_LEN, -1)
    reward_batch = torch.cat(episode_batch.reward).view(BATCH_SIZE, EPISODE_LEN)
    next_state_batch = torch.vstack(updated_next_state_batch).view(BATCH_SIZE, EPISODE_LEN + 1, -1)

    # Compute q-values of the policy network on the next state batch
    next_state_q_values = policy_net(next_state_batch, policy_hidden_state)[0]
    state_q_values = next_state_q_values[:, :-1, :]
    next_state_q_values_policy = next_state_q_values[:, 1:, :]

    state_action_values = state_q_values.gather(2, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    next_state_q_values_target = target_net(next_state_batch, target_hidden_state)[0]

    # We want everything but the first q values of each sequence
    next_state_q_values_target = next_state_q_values_target[:, 1:, :]

    next_state_values_target = next_state_q_values_target.gather(2, torch.max(next_state_q_values_policy, 2)[1].unsqueeze(2))
    # squeeze the last dimension
    next_state_values_target = next_state_values_target.squeeze(2)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values_target * GAMMA) + reward_batch

    # Compute loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(2))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    loss_lst.append(loss.item())


# optimization step with a small correction
def optimize_model_regularization():
    global steps_done

    if len(memory) < 10:
        return

    if LOSS == "Huber":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.MSELoss()

    policy_hidden_state = policy_net.init_hidden(batch_size=BATCH_SIZE)
    target_hidden_state = target_net.init_hidden(batch_size=BATCH_SIZE)

    # sars_lsts = memory.sample()
    sars_lsts = memory.weighted_sample()
    sars_joint = [sars_tuple for sublist in sars_lsts for sars_tuple in sublist]

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Sars objects
    # to Sars object of batch-arrays.

    episode_batch = Sars(*zip(*sars_joint))
    # change the next state batch
    init_state = torch.tensor([0]*memory.get_state_dim(), dtype=torch.float)
    empty_st_lst = [init_state] * BATCH_SIZE
    updated_next_state_batch = sum([[empty_st_lst[i]] + list(episode_batch.next_state[EPISODE_LEN*i: EPISODE_LEN*i + EPISODE_LEN]) for i in range(BATCH_SIZE)], [])

    state_batch = torch.vstack(episode_batch.state).view(BATCH_SIZE, EPISODE_LEN, -1)
    action_batch = torch.cat(episode_batch.action).view(BATCH_SIZE, EPISODE_LEN, -1)
    reward_batch = torch.cat(episode_batch.reward).view(BATCH_SIZE, EPISODE_LEN)
    next_state_batch = torch.vstack(updated_next_state_batch).view(BATCH_SIZE, EPISODE_LEN + 1, -1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_q_values = policy_net(state_batch, policy_hidden_state)[0]
    state_action_values = state_q_values.gather(2, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    next_state_q_values = target_net(next_state_batch, target_hidden_state)[0]

    # We want everything but the first q values of each sequence
    next_state_q_values = next_state_q_values[:, 1:, :]

    next_state_values = next_state_q_values.max(2)[0].detach()

    # regularization
    T = steps_done / TEMPERATURE
    policy_probs = F.softmax(next_state_q_values / T, dim=2)
    policy_probs_ent = Categorical(probs=policy_probs).entropy()

    # Compute the expected Q values
    expected_state_action_values = reward_batch + (next_state_values * GAMMA) + (REG_COEFF * policy_probs_ent)

    # Compute loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(2))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    loss_lst.append(loss.item())


# training loop
def training():
    global loss_lst
    loss_lst = []

    early_stopping_count = 3
    policy_hidden_state = policy_net.init_hidden()

    for i_episode in range(EPISODE_COUNT):
        # Initialize the environment and state
        dist_sys.reset()
        # at the beginning of each training sequence we have to reset the LSTM's hidden states
        policy_hidden_state = policy_net.reset_hidden(policy_hidden_state)

        state = dist_sys.system.get_rnn_input(None)
        episode_sars_lst = []
        failures = 0
        for t in range(EPISODE_LEN):
            # Select and perform an action
            action, policy_hidden_state = select_action_boltz(state, policy_hidden_state)
            next_state, reward = dist_sys.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Store the sars in memory
            episode_sars_lst.append(Sars(state, action, next_state, reward))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model2()

            # count failures
            if reward == -1:
                failures += 1

        memory.push(episode_sars_lst, EPISODE_LEN - failures)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 10 == 0:
            if evaluation(EPISODE_EVAL_LEN, 5) < 0.03:
                early_stopping_count -= 1
                if early_stopping_count == 0:
                    return
            else:
                early_stopping_count = 3

        if VERBOSITY:
            print("Episode: ", i_episode)
            print("Failures: ", failures)
            print()

    # plt.plot(range(len(loss_lst)), loss_lst)
    # plt.show()


# evaluation loop
def evaluation(episode_len, episode_count):
    avg_fail_rate = 0
    policy_hidden_state = policy_net.init_hidden()

    for i_episode in range(episode_count):
        # Initialize the environment and state
        dist_sys.reset()
        # at the beginning of each eval sequence we have to reset the LSTM's hidden states
        policy_hidden_state = policy_net.reset_hidden(policy_hidden_state)

        state = dist_sys.system.get_rnn_input(None)
        failures = 0
        for t in range(episode_len):
            # Select and perform an action
            action, policy_hidden_state = select_action(state, policy_hidden_state, exploration=False)
            next_state, reward = dist_sys.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Move to the next state
            state = next_state

            # count failures
            if reward == -1:
                failures += 1

        avg_fail_rate += (failures/episode_len)

    avg_fail_rate /= episode_count
    print("Average fail rate: {}%".format(avg_fail_rate * 100))

    return avg_fail_rate


# full experiment
print("=========NEW EXPERIMENT=========")
print("EXPERIMENT PARAMETERS:")
print("Figure number: {}".format(FIGURE_NUMBER))
print("Test repetitions: {}".format(TEST_REPETITIONS))
print("Verbosity: {}".format(VERBOSITY))
print("===============================")
print("HYPERPARAMETERS:")
print("Batch size: {}".format(BATCH_SIZE))
print("Discount factor: {}".format(GAMMA))
print("Epsilon start: {}".format(EPS_START))
print("Epsilon end: {}".format(EPS_END))
print("Epsilon decay: {}".format(EPS_DECAY))
print("Target net update frequency: {}".format(TARGET_UPDATE))
print("RNN hidden dimension: {}".format(RNN_H_DIM))
print("MLP hidden dimension: {}".format(HIDDEN_DIM))
print("Learning rate: {}".format(LEARNING_RATE))
print("Replay memory weight: {}".format(RE_MEM_WEIGHT))
print("Loss function: {}".format(LOSS))
print("Initial temperature: {}".format(TEMPERATURE))
print("===============================")
print("TRAINING PARAMETERS:")
print("Number of episodes: {}".format(EPISODE_COUNT))
print("Episode length: {}".format(EPISODE_LEN))
print("===============================")
print("EVALUATION PARAMETERS:")
print("Number of episodes: {}".format(EPISODE_EVAL_COUNT))
print("Episode length: {} \n\n".format(EPISODE_EVAL_LEN))

if FIGURE_NUMBER == 1:
    dist_sys = permitted_experiment_setup()
elif FIGURE_NUMBER == 2:
    dist_sys = schedule_experiment_setup()
elif FIGURE_NUMBER == 3:
    dist_sys = cases_experiment_setup()
elif FIGURE_NUMBER == 4:
    dist_sys = choice_scc_experiment_setup()
elif FIGURE_NUMBER == 5:
    dist_sys = cycle_scc_experiment_setup()


avg_failure_rate = 0
for _ in range(TEST_REPETITIONS):
    tr_count = len(dist_sys.system.transitions)
    input_dim = len(dist_sys.system.states) * tr_count

    policy_net = DRQN(input_dim, RNN_H_DIM, HIDDEN_DIM, tr_count).to(device)
    target_net = DRQN(input_dim, RNN_H_DIM, HIDDEN_DIM, tr_count).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(10000)
    steps_done = 0

    training()
    avg_failure_rate += evaluation(EPISODE_EVAL_LEN, EPISODE_EVAL_COUNT)
    print('\n\n')

avg_failure_rate /= TEST_REPETITIONS
print("Experiment {} Failure Rate: {}%".format(FIGURE_NUMBER, avg_failure_rate * 100))
