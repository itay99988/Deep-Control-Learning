import sys
import subprocess
import time

RUN_CMD = ["python", "ppo_experiment.py"]
RUN_EVAL_CMD = ["python", "ppo_experiment.py", "--mode=test", "--actor_model=ppo_actor.pth"]
LOGS_PATH = './logs'


def run_experiment(params):
    # basic params edit
    full_params = BASIC_PARAMS.copy()
    for k, v in params.items():
        full_params[k] = v

    args = RUN_CMD + ["--" + str(key) + "=" + str(param) for key, param in list(full_params.items())]
    result_train = subprocess.run(args=args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    result_eval = subprocess.run(args=RUN_EVAL_CMD, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

    output_file_name = f'{LOGS_PATH}/{time.strftime("%d%m%Y-%H%M%S")}.txt'
    with open(output_file_name, 'w') as output_file:
        if result_train.returncode != 0:
            output_file.write(result_train.stderr)
        else:
            output_file.write(result_train.stdout)
            output_file.write("\n")
            output_file.write(result_eval.stdout)


BASIC_PARAMS = {
    'batch_timesteps': 5000,
    'episode_timesteps': 100,
    'gamma': 0.99,
    'iteration_updates': 10,
    'lr': 1e-2,
    'clip': 0.2,
    'total_timesteps': 2000000
}

run_experiment({})
