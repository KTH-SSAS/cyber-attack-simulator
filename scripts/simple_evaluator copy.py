import copy
import shelve
from itertools import product
from pathlib import Path
from typing import Dict

import cloudpickle
import ray
from tqdm import tqdm

from attack_simulator.env.env import register_rllib_env
from attack_simulator.rllib.custom_callback import AttackSimCallback
from attack_simulator.rllib.defender_policy import Defender

trainers = {"PPO": Defender}

attackers = ["random", "pathplanner", "depth-first", "breadth-first", "mixed"]


def add_attacker(config: Dict, attacker):
    new_config = copy.deepcopy(config)
    new_config["old_attacker"] = config["env_config"]["attacker"]
    new_config["env_config"]["attacker"] = attacker
    return new_config


def get_checkpoint(folder):
    checkpoint = next(reversed(sorted((folder.glob("checkpoint_*")))))
    if checkpoint:
        with open(folder / "params.pkl", "rb") as f:
            run_config = cloudpickle.load(f)
    return checkpoint, run_config


# @ray.remote
def run_eval(config: dict, checkpoint, trainer_name: str, sweep_id, num: int):
    # config["env_config"]["false_positive"] = fp
    # config["env_config"]["false_negative"] = fn
    # if filename:
    #    config["env_config"]["graph_config"]["filename"] = filename
    # config["seed"] = seed

    trainer = trainers[trainer_name](config)
    trainer.restore(checkpoint)
    # results = trainer.evaluate()
    print(config["env_config"]["attacker"], end=" ")
    print(config["old_attacker"])

    return {} | {"trial_id": f"{trainer_name}_{sweep_id}_{num:05}"} | {"config": config}


def main():
    # Config options to ovveride
    config = {
        "callbacks": AttackSimCallback,
        "framework": "torch",
        "env": register_rllib_env(),
        "num_workers": 0,
        "disable_env_checking": True,
        "evaluation_interval": 1,
        "evaluation_num_workers": 0,
        "evaluation_duration": 500,
        "log_level": "ERROR",
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
    }

    # fp_range = np.linspace(0, 1, 5)
    # fn_range = np.linspace(0, 1, 5)
    # seeds = range(1, 4)
    # filenames = [None]

    ray_results = Path.home() / "sentience/data/ray_results/"
    sweeps = ["05065"]
    local_mode = False

    ray.init(local_mode=local_mode)

    checkpoint_configs = [
        (algorithm, sweep_id, get_checkpoint(folder))
        for sweep_id in sweeps
        for algorithm in ray_results.iterdir()
        for folder in sorted((ray_results / algorithm).glob(f"*_{sweep_id}_*"))
    ]

    merged_configs = [
        (algorithm, sweep_id, checkpoint, run_config | config)
        for algorithm, sweep_id, (checkpoint, run_config) in checkpoint_configs
    ]

    configs_with_attackers = [
        (algorithm, sweep_id, checkpoint, add_attacker(run_config, attacker))
        for attacker, (algorithm, sweep_id, checkpoint, run_config) in product(
            attackers, merged_configs
        )
    ]

    trials = (
        run_eval(run_config, checkpoint, algorithm.name, sweep_id + "eval", i)
        for i, (algorithm, sweep_id, checkpoint, run_config) in enumerate(configs_with_attackers)
    )

    for result in tqdm(trials, total=len(configs_with_attackers)):
        trial_id = result["trial_id"]
        sweep = trial_id.split("_")[1]
        with shelve.open(f"eval_{sweep}_attackers", flag="c") as db:
            db[result["trial_id"]] = result


main()
