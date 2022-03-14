import sys
import subprocess
import time

RUN_CMD = ["python", "dqn_experiment.py"]
LOGS_PATH = './logs'


def run_experiment(params):
    # basic params edit
    full_params = BASIC_PARAMS.copy()
    for k, v in params.items():
        full_params[k] = v

    args = RUN_CMD + [str(param) for param in list(full_params.values())]
    result = subprocess.run(args=args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)

    output_file_name = f'{LOGS_PATH}/{time.strftime("%d%m%Y-%H%M%S")}.txt'
    with open(output_file_name, 'w') as output_file:
        if result.returncode != 0:
            output_file.write(result.stderr)
        else:
            output_file.write(result.stdout)


BASIC_PARAMS = {
            "FIGURE_NUMBER": 5,
            "TEST_REPETITIONS": 1,
            "VERBOSITY": 0,

            "BATCH_SIZE": 2,
            "GAMMA": 0.9,
            "EPS_START": 0.9,
            "EPS_END": 0.5,
            "EPS_DECAY": 200,
            "TARGET_UPDATE": 50,
            "RNN_H_DIM": 10,
            "HIDDEN_DIM": 7,
            "LEARNING_RATE": 0.01,
            "RE_MEM_WEIGHT": 4,
            "LOSS": "Huber",
            "TEMPERATURE": 8000,

            "EPISODE_COUNT": 2000,
            "EPISODE_LEN": 50,

            "EPISODE_EVAL_COUNT": 100,
            "EPISODE_EVAL_LEN": 200}


run_experiment({})