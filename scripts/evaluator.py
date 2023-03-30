#!/usr/bin/env python

import argparse
import re
from pathlib import Path

import ray

import attack_simulator.rllib.ids_model as ids_model
from attack_simulator.env.env import register_rllib_env
from attack_simulator.rllib import evaluate


def get_checkpoint_path(run_dir: Path) -> Path:
    checkpoint_folder = reversed(sorted((run_dir.glob("checkpoint_*")))).__next__()
    # for f in checkpoint_folder.glob("checkpoint-*"):
    # 	if re.match(r"checkpoint-\d\d?\d?$", f.name):
    # 		checkpoint = Path(f)
    # 		break
    return checkpoint_folder


def run_evaluation(
    run_id,
    parser: argparse.ArgumentParser,
    algorithm,
    env_name,
    num_episodes,
    checkpoint=None,
    config=None,
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

    ray_results = Path("/home/jakob/ray_results/")

    sweeps = []
    local_mode = False
    num_episodes = 500

    ray.init(local_mode=local_mode)
    for sweep_id in sweeps:
        for algorithm in ray_results.iterdir():
            result_folders = filter(
                lambda x: re.search(sweep_id, x.name), (ray_results / algorithm).iterdir()
            )
            for folder in result_folders:
                run_id = "_".join(folder.name.split("_")[2:4])
                run_id = "_".join([algorithm.name, run_id])
                checkpoint = get_checkpoint_path(folder)
                if checkpoint:
                    run_evaluation(
                        run_id,
                        parser,
                        algorithm.name,
                        env_name,
                        num_episodes,
                        checkpoint=str(checkpoint),
                    )


if __name__ == "__main__":
    main()
