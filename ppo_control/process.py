import random
import torch
from torch import nn

SUCCESS = "(Success)"
FAIL = "(Failure)"
RANDOM = "(Random)"


# A transition class (transition of a process). contains name, source state, target state, and a status (success,
# failure or random)
class Transition:
    def __init__(self, name, source_state, target_state, status=None):
        self.name = name
        self.source_state = source_state
        self.target_state = target_state
        self.status = status

    def copy(self, status):
        return Transition(self.name, self.source_state, self.target_state, status)

    # a string representation of a transition - to be used in the execution report
    def __str__(self):
        return "{0} ---------{1}{2}---------> {3}".format(self.source_state, self.name, self.status, self.target_state)


# This class contains the three elements that every item in the lookahead buffer should have for the loss calculation:
# the transition distribution according to the network at prediction, the available transitions at this time and the
# chosen transition.
class LookaheadItem:
    def __init__(self, distribution, available_tr_lst, chosen_tr):
        self.distribution = distribution
        self.available_tr_lst = available_tr_lst
        self.chosen_tr = chosen_tr

    def is_transition_failed(self):
        return self.chosen_tr.status == FAIL


# the buffer that contains all the lookahead items. it functions as a queue - the first lookahead item which was
# added is the first one to be popped out of the buffer.
class LookaheadBuffer:
    def __init__(self, buf_len):
        self.length = buf_len
        self.buffer = []

    # returns the desired len of the lookahead buffer
    def get_max_len(self):
        return self.length

    # returns the current length of the buffer (the buffer may be only partially full at the beginning of
    # training executions)
    def get_cur_len(self):
        return len(self.buffer)

    # add an item to the lookahead buffer. if the buffer exceed its limit - we remove the oldest item
    def add_item(self, distribution, available_tr_lst, chosen_tr):
        self.buffer.append(LookaheadItem(distribution, available_tr_lst, chosen_tr))

        delta_len = len(self.buffer) - self.length
        if delta_len > 0:
            self.buffer = self.buffer[delta_len:]

    def get_past_distribution(self):
        if len(self.buffer) > 0:
            return self.buffer[0].distribution
        return None

    def get_past_avail_tr_lst(self):
        if len(self.buffer) > 0:
            return self.buffer[0].available_tr_lst
        return None

    def get_past_chosen_tr(self):
        if len(self.buffer) > 0:
            return self.buffer[0].chosen_tr
        return None

    # returns the amount of failures in the lookahead (required for the loss function)
    def count_failures(self):
        fail_count = sum([1 for itm in self.buffer if itm.chosen_tr.status == FAIL])
        return fail_count


# A process class - can be either the system or the environment (a finite and deterministic automaton)
class Process:
    def __init__(self, name, states=[], transitions=[], initial_state=None):
        self.name = name
        self.states = states  # list of names of the states.
        self.initial_state = initial_state
        self.current_state = initial_state
        self.transitions = transitions  # transitions that are specific to the process.

    def add_state(self, name):
        self.states.append(name)
        if self.current_state is None:
            self.initial_state = name
            self.current_state = name

    # return the current state of the process
    def get_current_state(self):
        return self.current_state

    # sets a new state as the current state (in case of triggering a transition for example)
    def set_current_state(self, name):
        if name in self.states:
            self.current_state = name

    # adds a new transition to the process
    def add_transition(self, name, source, target):
        self.transitions.append(Transition(name, source, target))

    # returns to the initial state of the transition
    def reset(self):
        self.set_current_state(self.initial_state)

    # returns the correct transition object according to its name and its source state
    # (the source state is enough in order to distinguish two transitions with the same name)
    def get_transition(self, tr_name, source_state=None):
        if source_state is None:
            source_state = self.current_state
        possible_tr = (tr for tr in self.transitions if tr.name == tr_name and tr.source_state == source_state)
        return next(possible_tr)

    # returns the index of a specific transition from the transition list. important for the loss function calculation.
    def get_transition_idx(self, tr_name, source_state=None):
        if source_state is None:
            source_state = self.current_state
        possible_idx = (i for i, tr in enumerate(self.transitions) if tr.name == tr_name and tr.source_state == source_state)
        return next(possible_idx)

    # copies an entire transition object, with a new status
    def copy_transition_w_status(self, tr_name, source_state=None, status=None):
        if source_state is None:
            source_state = self.current_state
        orig_tr = self.get_transition(tr_name, source_state)
        return orig_tr.copy(status)

    # returns a set of names of transitions that can be triggered in the current state.
    def available_transitions(self):
        available = []
        for tr in self.transitions:
            if tr.source_state == self.current_state:
                available.append(tr.name)
        return available

    # switches the process' state according to the transition tr_name.
    def trigger_transition(self, tr_name):
        try:
            possible_next_states = (tr.target_state for tr in self.transitions
                                    if tr.name == tr_name and tr.source_state == self.current_state)
            self.current_state = next(possible_next_states)

        except StopIteration:
            print("No transition named", tr_name, "from state", self.current_state)

    # chooses a random available state uniformly
    def get_random_transition(self):
        return random.choice(self.available_transitions())

    # checks if a certain transition is currently enabled
    def is_transition_enabled(self, tr_name):
        return tr_name in self.available_transitions()

    # gets the last system's transition (and its status - success or fail) and returns the input of the controller
    # this input reprsents a flattened matrix (transitions X states) with only one non zero cell. the location of the
    # non zero cell depends of the last transition and the current state. the non zero cell will be "1" if the last
    # transition was successful and "-1" otherwise.
    def get_rnn_input(self, tr: Transition):
        vec = [0] * (len(self.transitions) * len(self.states))

        if tr is not None:
            state_idx = self.states.index(self.current_state)
            transition_idx = self.get_transition_idx(tr.name, tr.source_state)

            # "-1" in case of failure and "1" otherwise
            if tr.status == FAIL:
                vec[len(self.states) * transition_idx + state_idx] = -1
            else:
                vec[len(self.states) * transition_idx + state_idx] = 1

        return vec

    # returns the name of the predicted transition according to the network output
    # there are two different methods to infer the next transition according the network output:
    # 1. randomly choose a transition according the softmax distribution of the network
    # 2. always choose the most probable transition (argmax)
    def get_predicted_transition(self, rnn_output, method="by_tr_distribution", debug_prob=False, debug_file="sys1_probs"):
        available_transitions = self.available_transitions()
        # leave only the possible transitions
        distribution = [rnn_output[:, i] for i, tr in enumerate(self.transitions)
                        if tr.name in available_transitions
                        and tr.source_state == self.current_state]

        if debug_prob:
            with open(debug_file, 'a') as f:
                print(distribution, file=f)

        # randomly choose a transition according the softmax distribution of the network
        if method == "by_tr_distribution":
            predicted_transition = random.choices(available_transitions, distribution)[0]

        # always choose the most probable transition at the time
        elif method == "argmax":
            i2v = lambda i: distribution[i]
            idx_max = max(range(len(distribution)), key=i2v)
            predicted_transition = available_transitions[idx_max]

        else:
            raise NameError("Incorrect method")

        # return the chosen transition
        return predicted_transition
