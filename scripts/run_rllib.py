import argparse
import dataclasses
import os
import socket
from dataclasses import asdict
from datetime import datetime
from time import strftime
from typing import Tuple

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback

from attack_simulator.config import EnvConfig
from attack_simulator.custom_callback import AttackSimCallback
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.rng import set_seeds
from attack_simulator.telegram_utils import notify_ending


def add_fp_tp_sweep(config: dict, values: list) -> dict:
    config["env_config"]["false_negative"] = tune.grid_search(values)
    config["env_config"]["false_positive"] = tune.grid_search(values)
    return config

def add_graph_sweep(config: dict, values: list) -> dict:
    config["env_config"]["graph_config"]["filename"] = tune.grid_search(values)
    return config

def add_seed_sweep(config, values: list) -> dict:
    config["seed"] = tune.grid_search(values)
    return config

def add_dqn_options(config: dict) -> dict:
    return config | {
        "batch_mode": "complete_episodes",
        "noisy": True,
        "num_atoms": 5,
        "v_min": -6000.0,
        "v_max": 6000.0,
        "train_batch_size": 600,
    }

def add_ppo_options(config: dict) -> dict:
    return config | {
        "train_batch_size": 600
    }

def dict2choices(d: dict) -> Tuple[list, str]:
    choices = list(d.keys())
    choices_help = '", "'.join(choices[:-1]) + f'" or "{choices[-1]}'
    return choices, choices_help


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Reinforcement learning of a computer network defender, using RLlib"
    )

    parser.add_argument("-g", "--graph", action="store_true", help="Generate a GraphViz .dot file.")

    parser.add_argument("--config-file", type=str, help="Path to YAML configuration file.")

    parser.add_argument("--stop-iters", type=int, help="Number of iterations to train.")
    parser.add_argument("--stop-timesteps", type=int, help="Number of timesteps to train.")
    parser.add_argument("--stop-reward", type=float, help="Reward at which we stop training.")

    parser.add_argument("--eval-interval", type=int, default=50)

    parser.add_argument(
        "--render", action="store_true", help="Render an animation of the evaluation."
    )

    parser.add_argument("-L", "--lr", help="Optimizer learning rate.", default=1e-2)

    parser.add_argument("-C", "--cuda", action="store_true", help="Use CUDA acceleration.")

    parser.add_argument("--wandb-sync", action="store_true", help="Sync run with wandb cloud.")

    parser.add_argument("--checkpoint-path", type=str)

    parser.add_argument("--gpu-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12000)
    parser.add_argument("--local", action="store_true", help="Enable ray local mode for debugger.")

    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--env-per-worker", type=int)

    parser.add_argument("--run-type", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"

    if args.local:
        local = True
    else:
        local = False

    ray.init(dashboard_host=dashboard_host, local_mode=local, num_cpus=7)

    callbacks = []

    os.environ["WANDB_MODE"] = "online" if args.wandb_sync else "offline"

    wandb_api_file_exists = os.path.exists("./wandb_api_key")
    if wandb_api_file_exists or os.environ.get("WANDB_API_KEY") is not None:

        api_key_file = "./wandb_api_key" if wandb_api_file_exists else None

        callbacks.append(
            WandbLoggerCallback(
                project="rl_attack_sim",
                group=f"sweep_{strftime('%H:%M')}",
                api_key_file=api_key_file,
                log_config=False,
                entity="sentience",
                job_type=args.run_type,
            )
        )

    env_config = EnvConfig.from_yaml(args.config_file)

    current_time = datetime.now().strftime(r"%m-%d_%H:%M:%S")
    id_string = f"{current_time}@{socket.gethostname()}"
    env_config = dataclasses.replace(env_config, save_graphs=args.graph, run_id=id_string)

    #model_config = {"use_lstm": True, "lstm_cell_size": 256}

    gpu_count = args.gpu_count
    batch_size = args.batch_size
    num_workers = args.num_workers
    env_per_worker = args.env_per_worker

    # Set global seeds
    set_seeds(5)

    # Allocate GPU power to workers
    # This is optimized for a single machine with multiple CPU-cores and a single GPU
    num_gpus = 0.0001 if gpu_count > 0 else 0
    gpus_per_worker = (
        (gpu_count - num_gpus) / num_workers if num_workers > 0 and num_gpus > 0 else 0
    )

    # fragment_length = 200

    config = {
        "framework": "torch",
        "env": AttackSimulationEnv,
        # This is the fraction of the GPU(s) this trainer will use.
        "num_gpus": 0.15,
        "num_workers": num_workers,
        "num_envs_per_worker": env_per_worker,
        #"num_gpus_per_worker": gpus_per_worker,
        #"model": model_config,
        "env_config": asdict(env_config),
        "batch_mode": "complete_episodes",
        # The number of iterations between renderings
        "evaluation_interval": args.stop_iters,
        #"evaluation_num_episodes": 1,
        #"evaluation_num_workers": 0,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
           "render_env": True,
        },
        "callbacks": AttackSimCallback,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # Remove stop conditions that were not set
    keys = list(stop.keys())
    for k in keys:
        if stop[k] is None:
            del stop[k]
    
    trainer = "PPO"

    if trainer == "PPO":
        config = add_ppo_options(config)
    else:
        config = add_dqn_options(config)

    config = add_seed_sweep(config, [1, 2, 3, 4])

    analysis: tune.ExperimentAnalysis = tune.run(
        trainer,
        config=config,
        stop=stop,
        callbacks=callbacks,
        checkpoint_at_end=True,
        metric="episode_reward_mean",
        mode="max",
        keep_checkpoints_num=1,
        checkpoint_freq=1,
        checkpoint_score_attr="episode_reward_mean",
        restore=args.checkpoint_path,
        #resume="PROMPT",
    )

    notify_ending(f"Tune has finished running {len(analysis.trials)} trials.")

    # wandb.config.update(env_config_dict)
    # wandb.config.update(graph_config_dict)
    # wandb.config.update(model_config)
    # wandb.config.update(config)
    # wandb.config.update(stop)


if __name__ == "__main__":
    main()
