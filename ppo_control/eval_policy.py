import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from utils import EXP_PATH

"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without
	relying on ppo.py.
"""


def _log_summary(ep_len, ep_ret, ep_num):
    # Round decimal places for more aesthetic logging messages
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy, env):
    """
        Returns a generator to roll out each episode given a trained policy and
        environment to test on.
        Parameters:
            policy - The trained policy to test
            env - The environment to evaluate the policy on

        Return:
            A generator object rollout, or iterable, which will return the latest
            episodic length and return on each iteration of the generator.
    """
    # Rollout until user kills process
    for _ in range(10):
        env.reset()
        state = env.get_compound_state()

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0  # episodic length
        ep_ret = 0  # episodic return

        while t < 100:
            t += 1

            # Query deterministic action from policy and run it
            state = torch.tensor(state, dtype=torch.float)

            # select the best action
            available_actions_idx = [env.system.get_transition_idx(tr_name) for tr_name in
                                     env.system.available_transitions()]
            logits = policy(state)
            filtered_logits = logits[available_actions_idx]
            dist = Categorical(logits=filtered_logits)
            action_idx = dist.sample()
            real_action_idx = available_actions_idx[action_idx.numpy()]

            state, reward = env.step(real_action_idx)

            # Sum all episodic rewards as we go along
            ep_ret += reward

        # Track episodic length
        ep_len = t

        # Print experiment report
        env.export_results(EXP_PATH)
        # returns episodic length and return in this iteration
        yield ep_len, ep_ret


def eval_policy(policy, env):
    """
        The main function to evaluate our policy with. It will iterate a generator object
        "rollout", which will simulate each episode and return the most recent episode's
        length and return. We can then log it right after. And yes, eval_policy will run
        forever until you kill the process.
        Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
        Return:
            None
        NOTE: To learn more about generators, look at rollout's function description
    """
    # Rollout with the policy and environment, and log each episode's data
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
