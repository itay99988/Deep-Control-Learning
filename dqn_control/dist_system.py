import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from process import Transition, Process, LookaheadBuffer, SUCCESS, FAIL, RANDOM


# a distributed system is a compound of processes (of class "Process").
class DistSystem(object):
    def __init__(self, name, processes):
        self.name = name
        self.processes = processes

    # reinitialize all processes in the system
    def reset(self):
        for pr in self.processes:
            pr.reset()

    # returns the process with that name
    def get_process(self, name):
        return next(proc for proc in self.processes if name == proc.name)

    # adds a new process to the system
    def add_process(self, process):
        self.processes.append(process)

    # adds a new transition to all the processes
    def add_transition(self, name, pr_list, source_list, target_list):
        for i in range(len(pr_list)):
            source = source_list[i]
            target = target_list[i]
            self.get_process(pr_list[i]).add_transition(name, source, target)


# This class inherits the properties of DistSystem. It is a specific case in which there are only two processes:
# a system and an environment. the system is the only process allowed to propose transitions to the environment.
# the system process has an RNN-based controller that can learn what are the optimal transitions to propose to the
# environment at any timestep.
class BlackboxEnvironment(DistSystem):
    def __init__(self, name, system: Process, environment: Process):
        super().__init__(name, [system, environment])
        self.system = system
        self.environment = environment

    def step(self, sys_tr_idx):
        next_state = None
        sys_tr = self.system.transitions[sys_tr_idx].name

        if self.environment.is_transition_enabled(sys_tr):
            full_sys_tr = self.system.copy_transition_w_status(sys_tr, status=SUCCESS)

            # trigger the transition for both system and environment
            self.system.trigger_transition(sys_tr)
            self.environment.trigger_transition(sys_tr)
            reward = 1
        else:
            full_sys_tr = self.system.copy_transition_w_status(sys_tr, status=FAIL)
            env_rnd_tr = self.environment.get_random_transition()

            # trigger the randomly chosen transition of the environment
            self.environment.trigger_transition(env_rnd_tr)
            reward = -1

        next_state = self.system.get_rnn_input(full_sys_tr)
        return next_state, reward

    # this function returns a random execution of the interaction between the system and the environment.
    # in this case, the system randomly chooses the next transition to be proposed to the environment.
    def random_execution(self, steps):
        system_execution = []
        environment_execution = []

        failure_counter = 0
        self.reset()

        # it is important to first append to system execution and only then trigger.. (for copy_transition_w_status)
        for s in range(steps):
            # the system chooses a random transition
            sys_rnd_tr = self.system.get_random_transition()
            # the environment can accepted the offered transition
            if self.environment.is_transition_enabled(sys_rnd_tr):
                system_execution.append(self.system.copy_transition_w_status(sys_rnd_tr, status=SUCCESS))
                environment_execution.append(self.environment.copy_transition_w_status(sys_rnd_tr, status=SUCCESS))
                # trigger the transition for both system and environment
                self.system.trigger_transition(sys_rnd_tr)
                self.environment.trigger_transition(sys_rnd_tr)
            # the environment cannot accepted the offered transition
            else:
                # the environment has to choose a transition from its available transitions
                env_rnd_tr = self.environment.get_random_transition()
                system_execution.append(self.system.copy_transition_w_status(sys_rnd_tr, status=FAIL))
                environment_execution.append(self.environment.copy_transition_w_status(env_rnd_tr, status=RANDOM))
                # trigger the randomly chosen transition of the environment
                self.environment.trigger_transition(env_rnd_tr)
                failure_counter += 1

        fail_rate = failure_counter / steps
        return system_execution, environment_execution, fail_rate

    # in this case, the system's controller is gradually trained to achieve a lower failure rates. the weights update
    # happens after each interaction between the system and the environment. the loss function's value depends on
    # the last transition (success or failure)
    def single_training_execution(self, steps, lookahead=0, random_step=0, epsilon=0):
        lookahead_buffer = LookaheadBuffer(lookahead+1)
        system_execution = []
        environment_execution = []
        last_transition = None
        failure_counter = 0

        # initial states are set for both system and environment
        self.reset()
        # our model is in "training mode"
        self.model.train()

        # at the beginning of each training sequence we have to reset the LSTM's hidden states
        self.model_hidden_state, self.model_cell_state = self.model.reset_states(self.model_hidden_state, self.model_cell_state)

        for step in range(steps):
            # encode the input to the network based on the last transition and the current state
            rnn_input = self.system.get_rnn_input(last_transition)

            self.model_hidden_state, self.model_cell_state = (self.model_hidden_state.detach(), self.model_cell_state.detach())

            # feed the network with the new input
            tr_scores, self.model_hidden_state, self.model_cell_state = self.model(rnn_input, self.model_hidden_state, self.model_cell_state)

            # choosing the next transition: exploration / exploitation
            if step > random_step:
                # with a probability of epsilon: a random transition will be proposed by the system
                if random.random() < epsilon:
                    next_transition = self.system.get_random_transition()
                # otherwise, we will exploit the network: we will choose the next action based on the network's output
                else:
                    next_transition = self.system.get_predicted_transition(tr_scores, method="by_tr_distribution")
            else:
                next_transition = self.system.get_random_transition()

            # we need to check if the environment can accept next_transition
            tr_success = self.environment.is_transition_enabled(next_transition)

            if tr_success:
                system_execution.append(self.system.copy_transition_w_status(next_transition, status=SUCCESS))
                environment_execution.append(self.environment.copy_transition_w_status(next_transition, status=SUCCESS))
                # update the lookahead buffer
                lookahead_buffer.add_item(tr_scores, self.system.available_transitions(), system_execution[-1])
                self.system.trigger_transition(next_transition)
                self.environment.trigger_transition(next_transition)
            else:
                next_env_transition = self.environment.get_random_transition()
                system_execution.append(self.system.copy_transition_w_status(next_transition, status=FAIL))
                environment_execution.append(self.environment.copy_transition_w_status(next_env_transition, status=RANDOM))
                # update the lookahead buffer
                lookahead_buffer.add_item(tr_scores, self.system.available_transitions(), system_execution[-1])
                self.environment.trigger_transition(next_env_transition)
                failure_counter += 1

            # keep the last system's transition. this will be used in the next iteration to encode the network's input.
            last_transition = system_execution[-1]

            # we first calculate the loss based on the lookahead buffer, then we clear optimizer's memory regarding
            # the previous gradient calculations. Finally we call backward() to calculate the new gradients
            # based on the loss function, and update the model's weight using the step() function.
            if step > random_step:
                loss = self.reinforce_loss(lookahead_buffer)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

        fail_rate = failure_counter / steps
        return system_execution, environment_execution, fail_rate

    # "exploit" the trained model. in this kind of execution the system will be based solely on the trained rnn as
    # the controller.
    def single_controlled_execution(self, steps, debug_prob=False):
        system_execution = []
        environment_execution = []
        last_transition = None
        failure_counter = 0

        self.reset()
        # our model is in "training mode"
        self.model.eval()

        self.model_hidden_state, self.model_cell_state = self.model.reset_states(self.model_hidden_state,
                                                                                 self.model_cell_state)

        for step in range(steps):
            rnn_input = self.system.get_rnn_input(last_transition)
            self.model_hidden_state, self.model_cell_state = (self.model_hidden_state.detach(), self.model_cell_state.detach())

            tr_scores, self.model_hidden_state, self.model_cell_state = self.model(rnn_input, self.model_hidden_state, self.model_cell_state)

            # in this case we always exploit the model, since we assume it is fully trained
            next_transition = self.system.get_predicted_transition(tr_scores, method="argmax", debug_prob=debug_prob)

            # we need to check if the environment can accept next_transition
            tr_success = self.environment.is_transition_enabled(next_transition)
            if tr_success:
                system_execution.append(self.system.copy_transition_w_status(next_transition, status=SUCCESS))
                environment_execution.append(self.environment.copy_transition_w_status(next_transition, status=SUCCESS))
                self.system.trigger_transition(next_transition)
                self.environment.trigger_transition(next_transition)
            else:
                next_env_transition = self.environment.get_random_transition()
                system_execution.append(self.system.copy_transition_w_status(next_transition, status=FAIL))
                environment_execution.append(self.environment.copy_transition_w_status(next_env_transition, status=RANDOM))
                self.environment.trigger_transition(next_env_transition)
                failure_counter += 1

            # keep the last system's transition
            last_transition = system_execution[-1]

        fail_rate = failure_counter / steps
        return system_execution, environment_execution, fail_rate
