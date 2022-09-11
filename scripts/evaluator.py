#!/usr/bin/env python

import argparse
from pathlib import Path
from attack_simulator import evaluate
from attack_simulator.env import register_rllib_env
import attack_simulator.ids_model as ids_model
from tqdm import tqdm
import re
import ray

from ray.rllib.algorithms.registry import ALGORITHMS

def _import_tripwire():
    import attack_simulator.optimal_defender as optimal_defender
    return optimal_defender.TripwireDefender, {}

def _import_random():
    import attack_simulator.random_defender as random_defender
    return random_defender.RandomDefender, {}

ALGORITHMS["Tripwire"] = _import_tripwire
ALGORITHMS["Random"] = _import_random

def has_checkpoint(algorithm: str):
    return algorithm in ["PPO", "DQN"]


def get_checkpoint_path(run_dir: Path) -> Path:
    checkpoint_folder = reversed(sorted((run_dir.glob("checkpoint_*")))).__next__()
    # for f in checkpoint_folder.glob("checkpoint-*"):
    # 	if re.match(r"checkpoint-\d\d?\d?$", f.name):
    # 		checkpoint = Path(f)
    # 		break
    return checkpoint_folder


def run_evaluation(run_id, 
    parser: argparse.ArgumentParser, algorithm, env_name, num_episodes, checkpoint=None, config=None
):

    args_to_parse = [
        "--id",
        run_id,
        "--out",
        "eval_data",
        "--env",
        env_name,
        "--run",
        algorithm,
        "--episodes",
        str(num_episodes),
        "--steps",
        "0",  # No limit on amount of steps
        "--use-shelve",
    ]

    if config:
        args_to_parse.extend(["--config_path", config])

    if checkpoint:
        args_to_parse = [checkpoint] + args_to_parse

    args = parser.parse_args(args_to_parse)
    evaluate.run(args, parser)


def main():
    env_name = register_rllib_env()
    ids_model.register_rllib_model()

    parser = evaluate.create_parser()

    ray_results = Path("/home/jakob/sentience/data/ray_results/")

    sweeps = ["0a2b5"]
    local_mode = True
    num_episodes = 500

    ray.init(local_mode=local_mode)
    for sweep_id in sweeps:
        for algorithm in ray_results.iterdir():
            result_folders = filter(lambda x: re.search(sweep_id, x.name), (ray_results / algorithm).iterdir())
            for folder in result_folders:
                run_id = "_".join(folder.name.split('_')[2:4])
                run_id = "_".join([algorithm.name, run_id])
                if not has_checkpoint(algorithm.name):
                    config_file = folder / "params.pkl"
                    run_evaluation(run_id, parser, algorithm.name, env_name, num_episodes, config=str(config_file))
                else:
                    checkpoint = get_checkpoint_path(folder)
                    if checkpoint:
                        run_evaluation(run_id, parser, algorithm.name, env_name, num_episodes, checkpoint=checkpoint)


if __name__ == "__main__":
    main()
