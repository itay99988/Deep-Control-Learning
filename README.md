
# Deep Reinforcement Learning of Sythesizing a Control for a Blackbox Environment
This repository contains two directories, one for each tested approach: 
1. Deep Recurrent Q-Network
2. Proximal Policy Optimization

**Please note: pytorch package is required.**

## Deep Recurrent Q-Network
The repository contains the following files:
- bulk_run.py - runs a single experiment with the chosen hyperparameters
- dist_system.py - the main purpose of this file is to model the interactions between the system and the environment
- dqn_control.py - the network architecture
- dqn_experiment.py - the algorithm itself with all the added modifications mentioned in the report.
- experiment.py - the experiments of the paper, as python objects
- process.py - a class of automaton - used by dist_system
- utils.py - prints an execution report of a single controlled execution

Execution instructions:
- In the file bulk_run.py, change the variable RUN_CMD according to the environment variable in your local environment.
- Assign the desired values to the hyperparameters in the file bulk_run.py.
  if you want to repeat an experiment, change the value of the "repetitions" parameters.
- For different experiment types please see the following:
"FIGURE_NUMBER" = 1            -> 'permitted' experiment
"FIGURE_NUMBER" = 2            -> 'schedule' experiment
"FIGURE_NUMBER" = 3            -> 'cases' experiment
"FIGURE_NUMBER" = 4            -> 'choice_scc' experiment
"FIGURE_NUMBER" = 5            -> 'cycle_scc' experiment
- run the file 'bulk_run.py' without command line arguments.
- Once the experiment is finished, a new log will be created in 'logs' directory.


## Proximal Policy Optimization
The repository contains the following files:
- arguments.py - contains the arguments to parse at command line. We won't be using it directly.
- network.py - This file contains a neural network module for us to define our actor and critic networks in PPO.
- ppo.py - The file contains the PPO class to train with.
- ppo_experiment.py - contains the main training and test functions of the model.
- bulk_run.py - runs a single experiment with the chosen hyperparameters
- dist_system.py - the main purpose of this file is to model the interactions between the system and the environment
- experiment.py - the experiments of the paper, as python objects
- process.py - a class of automaton - used by dist_system
- utils.py - prints an execution report of a single controlled execution

Execution instructions:
- In the file bulk_run.py, change the variable RUN_CMD according to the environment variable in your local environment.
- Assign the desired values to the hyperparameters in the file bulk_run.py.
- run the file 'bulk_run.py' without command line arguments.
- Once the experiment is finished, a new log will be created in 'logs' directory.
