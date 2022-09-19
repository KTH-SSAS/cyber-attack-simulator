import argparse
import dataclasses
import os
import socket
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Tuple

import ray
from ray import tune
from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune.utils.log import Verbosity
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer

from attack_simulator.config import EnvConfig
from attack_simulator.custom_callback import AttackSimCallback
from attack_simulator.env import AttackSimulationEnv, register_rllib_env
import attack_simulator.ids_model as ids_model
import attack_simulator.optimal_defender as optimal_defender
import attack_simulator.random_defender as random_defender
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
    return config | {
        "train_batch_size": 4096,
        "sgd_minibatch_size": 256,
        "vf_clip_param": 500.0,
        "clip_param": 0.05,
        "vf_loss_coeff": 0.001,
        "lr": 0.00001,
    }


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
    render: bool = False,
    **kwargs,
) -> None:
    """Main function for running the RLlib experiment."""

    dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"

    ray.init(dashboard_host=dashboard_host, local_mode=local_mode, num_cpus=7)

    callbacks = []

    os.environ["WANDB_MODE"] = "online" if wandb_sync else "offline"

    register_rllib_env()
    ids_model.register_rllib_model()

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

    model_config = {'fcnet_hiddens': [32, 32], "custom_model": "DefenderModel", "custom_model_config": {}}

    gpu_count = kwargs["gpu_count"]
    num_workers = kwargs["num_workers"]
    env_per_worker = kwargs["env_per_worker"]


    global_seed = 22
    # Set global seeds
    set_seeds(global_seed)

    # Allocate GPU power to workers
    # This is optimized for a single machine with multiple CPU-cores and a single GPU
    gpu_use_percentage = 0.15 if gpu_count > 0 else 0
    num_parallell_tasks = 3
    num_gpus = 0.001  # Driver GPU
    num_gpus_per_worker = (gpu_count / num_parallell_tasks - num_gpus) / num_workers
    # fragment_length = 200

    config = {
        "seed": global_seed,
        "horizon": 5000,
        "framework": "torch",
        "env": "AttackSimulationEnv",
        # This is the fraction of the GPU(s) this trainer will use.
        "num_gpus": 0 if local_mode else num_gpus,
        "num_workers": 0 if local_mode else 1,
        "num_envs_per_worker": 1 if local_mode else env_per_worker,
        "num_gpus_per_worker": num_gpus_per_worker,
        "model": model_config,
        "env_config": asdict(env_config),
        "batch_mode": "complete_episodes",
        # The number of iterations between renderings
        "evaluation_interval": stop_iterations,
        "evaluation_duration": 500,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            "render_env": render,
            "num_envs_per_worker": 1,
            "env_config": {
                "save_graphs": render,
                "save_logs": render,
            },
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

    config = add_fp_tp_sweep(config, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    run_ppo = True
    run_dqn = False
    run_random = True
    run_tripwire = True

    experiments = []

    if run_ppo:
        experiments.append(
            tune.Experiment(
                "PPO",
                PPOTrainer,
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
                DQNTrainer,
                config=add_dqn_options(config),
                stop=stop,
                checkpoint_at_end=True,
                keep_checkpoints_num=1,
                checkpoint_freq=1,
                checkpoint_score_attr="episode_reward_mean",
            )
        )

    heuristic_config = {
        "train_batch_size": 15,
        "num_gpus": 0,
        "simple_optimizer": True,
        "evaluation_interval": 1,
    }

    if run_random:
        experiments.append(
            tune.Experiment(
                "Random",
                random_defender.RandomDefender,
                config=config | heuristic_config,
                stop={"training_iteration": 1},
                checkpoint_score_attr="episode_reward_mean",
            )
        )

    if run_tripwire:
        dummy_env = AttackSimulationEnv(env_config)
        experiments.append(
            tune.Experiment(
                "Tripwire",
                optimal_defender.TripwireDefender,
                config=config
                | heuristic_config
                | {
                    "defense_steps": dummy_env.sim.g.attack_steps_by_defense_step,
                },
                stop={"training_iteration": 1},
                checkpoint_score_attr="episode_reward_mean",
            )
        )

    analysis: tune.ExperimentAnalysis = tune.run_experiments(
        experiments,
        callbacks=callbacks,
        progress_reporter=tune.CLIReporter(max_report_frequency=60),
        verbose=Verbosity.V1_EXPERIMENT,
        # restore=args.checkpoint_path,
        # resume="PROMPT",
    )

    notify_ending(f"Run {id_string} finished.")

    # wandb.config.update(env_config_dict)
    # wandb.config.update(graph_config_dict)
    # wandb.config.update(model_config)
    # wandb.config.update(config)
    # wandb.config.update(stop)


if __name__ == "__main__":
    main(**parse_args())
