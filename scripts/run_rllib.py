import argparse
import dataclasses
import os
import socket
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback

from attack_simulator.config import EnvConfig
from attack_simulator.custom_callback import AttackSimCallback
from attack_simulator.env import AttackSimulationEnv, register_rllib_env
from attack_simulator.ids_model import register_rllib_model
from attack_simulator.rng import set_seeds
from attack_simulator.telegram_utils import notify_ending


def add_fp_tp_sweep(config: dict, values: list) -> dict:
    config["env_config"]["false_negative"] = tune.grid_search(values)
    config["env_config"]["false_positive"] = tune.grid_search(values)
    return config


def add_graph_sweep(config: dict, values: list) -> dict:
    config["env_config"]["graph_config"]["filename"] = tune.grid_search(values)
    return config


def add_seed_sweep(config: dict, values: list) -> dict:
    config["seed"] = tune.grid_search(values)
    return config


def add_dqn_options(config: dict) -> dict:
    return config | {
        "noisy": True,
        "num_atoms": 5,
        "v_min": -150.0,
        "v_max": 0.0,
        "train_batch_size": 600,
    }


def add_ppo_options(config: dict) -> dict:
    return config | {"train_batch_size": 600}


def dict2choices(d: dict) -> Tuple[list, str]:
    choices = list(d.keys())
    choices_help = '", "'.join(choices[:-1]) + f'" or "{choices[-1]}'
    return choices, choices_help


def parse_args() -> Dict[str, Any]:

    parser = argparse.ArgumentParser(
        description="Reinforcement learning of a computer network defender, using RLlib"
    )

    parser.add_argument("--config-file", type=str, help="Path to YAML configuration file.")

    parser.add_argument(
        "--stop-iters", type=int, help="Number of iterations to train.", dest="stop_iterations"
    )
    parser.add_argument("--stop-reward", type=float, help="Reward at which we stop training.")

    parser.add_argument("--eval-interval", type=int, default=50)

    parser.add_argument("-L", "--lr", help="Optimizer learning rate.", default=1e-2)

    parser.add_argument("-C", "--cuda", action="store_true", help="Use CUDA acceleration.")

    parser.add_argument("--wandb-sync", action="store_true", help="Sync run with wandb cloud.")

    parser.add_argument("--checkpoint-path", type=str)

    parser.add_argument("--gpu-count", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12000)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Enable ray local mode for debugger.",
        dest="local_mode",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--env-per-worker", type=int, default=1)

    parser.add_argument("--run-type", type=str, default=None)

    return vars(parser.parse_args())


def main(
    config_file: str,
    stop_iterations: int,
    local_mode: bool = False,
    wandb_sync: bool = False,
    **kwargs,
) -> None:
    """Main function for running the RLlib experiment."""

    dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"

    ray.init(dashboard_host=dashboard_host, local_mode=local_mode, num_cpus=7)

    callbacks = []

    os.environ["WANDB_MODE"] = "online" if wandb_sync else "offline"

    register_rllib_env()
    register_rllib_model()

    current_time = datetime.now().strftime(r"%m-%d_%H:%M:%S")
    id_string = f"{current_time}@{socket.gethostname()}"
    wandb_api_file_exists = os.path.exists("./wandb_api_key")
    if wandb_api_file_exists or os.environ.get("WANDB_API_KEY") is not None:

        api_key_file = "./wandb_api_key" if wandb_api_file_exists else None
        callbacks.append(
            WandbLoggerCallback(
                project="rl_attack_sim",
                group=f"Sweep_{id_string}",
                api_key_file=api_key_file,
                log_config=False,
                entity="sentience",
                tags=["train"],
            )
        )

    env_config = EnvConfig.from_yaml(config_file)
    env_config = dataclasses.replace(env_config, run_id=id_string)

    # To get the number of defenses
    test_env = AttackSimulationEnv(env_config)

    model_config = {
        "custom_model": "DefenderModel",
        "custom_model_config": {
            "num_defense_steps": test_env.sim.num_defense_steps,
        },
    }

    gpu_count = kwargs["gpu_count"]
    num_workers = kwargs["num_workers"]
    env_per_worker = kwargs["env_per_worker"]

    # Set global seeds
    set_seeds(5)

    # Allocate GPU power to workers
    # This is optimized for a single machine with multiple CPU-cores and a single GPU
    gpu_use_percentage = 0.15 if gpu_count > 0 else 0

    # fragment_length = 200

    config = {
        "horizon": 5000,
        "framework": "torch",
        "env": "AttackSimulationEnv",
        # This is the fraction of the GPU(s) this trainer will use.
        "num_gpus": 0 if local_mode else gpu_use_percentage,
        "num_workers": 0 if local_mode else num_workers,
        "num_envs_per_worker": 1 if local_mode else env_per_worker,
        # "num_gpus_per_worker": gpus_per_worker,
        "model": model_config,
        "env_config": asdict(env_config),
        "batch_mode": "complete_episodes",
        # The number of iterations between renderings
        "evaluation_interval": stop_iterations,
        "evaluation_num_episodes": 5,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            "render_env": True,
            "num_envs_per_worker": 1,
            "env_config": asdict(dataclasses.replace(env_config, save_graphs=True, save_logs=True)),
        },
        "callbacks": AttackSimCallback,
    }

    stop = {"training_iteration": stop_iterations, "episode_reward_mean": kwargs["stop_reward"]}

    # Remove stop conditions that were not set
    keys = list(stop.keys())
    for k in keys:
        if stop[k] is None:
            del stop[k]

    config = add_seed_sweep(config, [1, 2, 3])

    run_ppo = True
    run_dqn = False

    experiments = []

    if run_ppo:
        experiments.append(
            tune.Experiment(
                "PPO",
                run="PPO",
                config=add_ppo_options(config),
                stop=stop,
                checkpoint_at_end=True,
                keep_checkpoints_num=1,
                checkpoint_freq=1,
                checkpoint_score_attr="episode_reward_mean",
            )
        )

    if run_dqn:
        experiments.append(
            tune.Experiment(
                "DQN",
                run="DQN",
                config=add_dqn_options(config),
                stop=stop,
                checkpoint_at_end=True,
                keep_checkpoints_num=1,
                checkpoint_freq=1,
                checkpoint_score_attr="episode_reward_mean",
            )
        )

    analysis: tune.ExperimentAnalysis = tune.run_experiments(
        experiments,
        callbacks=callbacks,
        # restore=args.checkpoint_path,
        # resume="PROMPT",
    )

    if isinstance(analysis.trials, List):
        notify_ending(f"Tune has finished running {len(analysis.trials)} trials.")

    # wandb.config.update(env_config_dict)
    # wandb.config.update(graph_config_dict)
    # wandb.config.update(model_config)
    # wandb.config.update(config)
    # wandb.config.update(stop)


if __name__ == "__main__":
    main(**parse_args())
