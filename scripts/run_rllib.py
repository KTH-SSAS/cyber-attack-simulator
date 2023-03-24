import argparse
import copy
import dataclasses
import os
import random
import re
import socket
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Tuple
from pathlib import Path

import ray
import ray.train.torch
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.policy.policy import PolicySpec
from ray.tune.schedulers.pbt import PopulationBasedTraining

import attack_simulator.ids_model as ids_model
from attack_simulator.agents.attackers_policies import BreadthFirstPolicy
from attack_simulator.config import EnvConfig
from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.custom_callback import AttackSimCallback
from attack_simulator.env import AttackSimulationEnv, register_rllib_env


def add_fp_tp_sweep(config: dict, values: list) -> dict:
    new_config = copy.deepcopy(config)
    new_config["env_config"]["sim_config"]["false_negative_rate"] = tune.grid_search(values)
    new_config["env_config"]["sim_config"]["false_positive_rate"] = tune.grid_search(values)
    return new_config


def set_fp_fn_vals(config: dict, fp, fn):
    new_config = copy.deepcopy(config)
    new_config["env_config"]["sim_config"]["false_negative_rate"] = fn
    new_config["env_config"]["sim_config"]["false_positive_rate"] = fp
    return new_config


def add_graph_sweep(config: dict, values: list) -> dict:
    new_config = copy.deepcopy(config)
    new_config["env_config"]["graph_config"]["filename"] = tune.grid_search(values)
    return new_config


def add_attacker_sweep(config: dict, values: list) -> dict:
    new_config = copy.deepcopy(config)
    new_config["env_config"]["attacker"] = tune.grid_search(values)
    return new_config


def add_seed_sweep(config: dict, values: list) -> dict:
    new_config = copy.deepcopy(config)
    new_config["seed"] = tune.grid_search(values)
    return new_config


def add_dqn_options(config: dict) -> dict:
    new_config = copy.deepcopy(config)
    return new_config | {
        "noisy": True,
        "num_atoms": 5,
        "v_min": -150.0,
        "v_max": 0.0,
        "train_batch_size": 600,
    }


def get_graph_size(config):
    return int(re.search(r"_(\d+)\.yaml", config["env_config"]["graph_config"]["filename"])[1])


def add_ppo_options(config: dict) -> dict:
    new_config = copy.deepcopy(config)
    return new_config | {
        "train_batch_size": 2048,
        "sgd_minibatch_size": 2048,
        "vf_clip_param": 500,
        "clip_param": 0.02,
        "vf_loss_coeff": 0.001,
        "lr": 0.0001,
        "gamma": 1,
        "use_critic": True,
        "use_gae": True,
        "kl_coeff": 0.0,
        "entropy_coeff": 0.0,
        "scale_rewards": False,
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

    ray.init(dashboard_host=dashboard_host, local_mode=local_mode)

    callbacks = []

    os.environ["WANDB_MODE"] = "online" if wandb_sync else "offline"

    env_name = register_rllib_env()

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

    cwd = os.getcwd()
    absolute_graph_pathname = Path.absolute(Path(cwd, env_config.graph_config.filename))
    env_config = dataclasses.replace(
        env_config,
        graph_config=dataclasses.replace(
            env_config.graph_config, filename=str(absolute_graph_pathname)
        ),
    )
    env_config = dataclasses.replace(env_config, run_id=id_string)
    dummy_env = AttackSimulationEnv(env_config)
    env_config = dataclasses.replace(env_config, backend=tune.grid_search(["rust", "python"]))


    gpu_count = kwargs["gpu_count"]
    num_workers = kwargs["num_workers"]
    envs_per_worker = kwargs["env_per_worker"]

    global_seed = 22

    # Allocate GPU power to workers
    # This is optimized for a single machine with multiple CPU-cores and a single GPU
    # gpu_use_percentage = 0.15 if gpu_count > 0 else 0
    # num_parallell_tasks = 3
    # num_gpus = 0.001  # Driver GPU
    # num_gpus_per_worker = (gpu_count / num_parallell_tasks - num_gpus) / num_workers

    num_gpus = 0.0001 if gpu_count > 0 else 0  # Driver GPU
    num_gpus_per_worker = (
        (gpu_count - num_gpus) / num_workers if num_workers > 0 and gpu_count > 0 else 0
    )

    # fragment_length = 200

    attacker_policy_class = BreadthFirstPolicy

    config = (
        ids_model.DefenderConfig()
        .training(
            scale_rewards=False,
            gamma=1.0,
            sgd_minibatch_size=128,
            train_batch_size=128,
            vf_clip_param=500,
            clip_param=0.02,
            vf_loss_coeff=0.001,
            lr=0.0001,
            use_critic=True,
            use_gae=True,
            kl_coeff=0.0,
            entropy_coeff=0.0,
        )
        .framework("torch")
        .environment(env_name, env_config=asdict(env_config))
        .debugging(seed=global_seed)
        .resources(num_gpus=num_gpus, num_gpus_per_worker=num_gpus_per_worker)
        .evaluation(
            evaluation_interval=stop_iterations,
            evaluation_duration=500,
            evaluation_config={
                "render_env": render,
                "num_envs_per_worker": 1,
                "env_config": {
                    "save_graphs": render,
                    "save_logs": render,
                },
            },
        )
        .callbacks(AttackSimCallback)
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=envs_per_worker,
            batch_mode="complete_episodes",
        )
        .multi_agent(
            policies={
                AGENT_DEFENDER: PolicySpec(
                    policy_class=ids_model.DefenderPolicy,
                    config={
                        "model": {
                            "custom_model": "DefenderModel",
                            "fcnet_hiddens": [8],
                            "vf_share_layers": True,
                            "custom_model_config": {},
                        }
                    },
                ),
                AGENT_ATTACKER: PolicySpec(
                    attacker_policy_class,
                    config={
                        "num_special_actions": dummy_env.num_special_actions,
                        "wait_action": dummy_env.sim.wait_action,
                        "terminate_action": dummy_env.sim.terminate_action,
                    },
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            policies_to_train=[AGENT_DEFENDER],
        )
    )

    # Remove stop conditions that were not set
    # keys = list(stop.keys())
    # for k in keys:
    #     if stop[k] is None:
    #         del stop[k]

    # config = add_seed_sweep(config, [1])

    # config = add_graph_sweep(config, [
    #      f"graphs/second_graph_sweep/model_graph_{size}.yaml"
    #      for size in [20] #[80, 20, 60, 40]
    # ])

    # config = add_attacker_sweep(config, ["random", "pathplanner", "depth-first", "breadth-first", "mixed"])

    # config = add_fp_tp_sweep(config, np.geomspace(0.5, 1, 5)-0.5)

    # run_ppo = True
    # run_dqn = False

    # experiments = []

    # if run_ppo:
    #     experiments.append(
    #         tune.Experiment(
    #             "PPO",
    #             ids_model.Defender,
    #             config=add_ppo_options(config),
    #             stop=stop,
    #             checkpoint_at_end=True,
    #             keep_checkpoints_num=1,
    #             checkpoint_freq=1,
    #             checkpoint_score_attr="episode_reward_mean",
    #         )
    #     )

    # if run_dqn:
    #     experiments.append(
    #         tune.Experiment(
    #             "DQN",
    #             DQNTrainer,
    #             config=add_dqn_options(config),
    #             stop=stop,
    #             checkpoint_at_end=True,
    #             keep_checkpoints_num=1,
    #             checkpoint_freq=1,
    #             checkpoint_score_attr="episode_reward_mean",
    #         )
    #     )

    # heuristic_config = {
    #     "train_batch_size": 15,
    #     "num_gpus": 0,
    #     "simple_optimizer": True,
    #     "evaluation_interval": 1,
    # }

    # if run_random:
    #     experiments.append(
    #         tune.Experiment(
    #             "Random",
    #             random_defender.RandomDefender,
    #             config=config | heuristic_config,
    #             stop={"training_iteration": 1},
    #             checkpoint_score_attr="episode_reward_mean",
    #         )
    #     )

    # if run_tripwire:
    #     dummy_env = AttackSimulationEnv(env_config)
    #     experiments.append(
    #         tune.Experiment(
    #             "Tripwire",
    #             optimal_defender.TripwireDefender,
    #             config=config
    #             | heuristic_config
    #             | {
    #                 "defense_steps": dummy_env.sim.g.attack_steps_by_defense_step,
    #             },
    #             stop={"training_iteration": 1},
    #             checkpoint_score_attr="episode_reward_mean",
    #         )
    #      )

    # Hyperparameter tuning

    criteria = "episode_reward_mean"
    perturb = 0.25

    # pb2 = PB2(
    #     time_attr=criteria,
    #     metric="episode_reward_mean",
    #     mode="max",
    #     perturbation_interval=t_ready,
    #     quantile_fraction=perturb,  # copy bottom % with top %
    #     # Specifies the hyperparam search space
    #     hyperparam_bounds={
    #         "lambda": [0.9, 1.0],
    #         "clip_param": [0.1, 0.5],
    #         "lr": [1e-3, 1e-5],
    #         "train_batch_size": [1000, 60000],
    #     },
    # )

    # Postprocess the perturbed config to ensure it's still valid used if PBT.
    def explore(config):
        # Ensure we collect enough timesteps to do sgd.
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # Ensure we run at least one sgd iter.
        if config["lambda"] > 1:
            config["lambda"] = 1
        config["train_batch_size"] = int(config["train_batch_size"])
        return config

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=50,
        resample_probability=perturb,
        quantile_fraction=perturb,  # copy bottom % with top %
        # Specifies the search space for these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": lambda: random.uniform(1e-3, 1e-5),
            "train_batch_size": lambda: random.randint(300, 3000),
            "vf_clip_param": lambda: random.randint(100, 1000),
            "vf_loss_coeff": lambda: random.uniform(0.0001, 0.01),
        },
        custom_explore_fn=explore,
    )

    #     "train_batch_size": 2048,
    # "sgd_minibatch_size": 2048,
    # "vf_clip_param": 500,
    # "clip_param": 0.02,
    # "vf_loss_coeff": 0.001,
    # "lr": 0.0001,
    # "gamma": 1,
    # "use_critic": True,
    # "use_gae": True,
    # "kl_coeff": 0.0,
    # "entropy_coeff": 0.0,
    # "scale_rewards": False,

    tuner = tune.Tuner(
        ids_model.Defender,
        tune_config=tune.TuneConfig(reuse_actors=False),
        run_config=air.RunConfig(
            stop=tune.stopper.MaximumIterationStopper(stop_iterations),
            callbacks=callbacks,
            # progress_reporter=tune.CLIReporter(max_report_frequency=60),
        ),
        param_space=config,
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    print("Best result:", best_result)

    # analysis: tune.ExperimentAnalysis = tune.run_experiments(
    #     experiments,
    #     callbacks=callbacks,
    #     progress_reporter=tune.CLIReporter(max_report_frequency=60),
    #     verbose=Verbosity.V1_EXPERIMENT,
    #     # restore=args.checkpoint_path,
    #     # resume="PROMPT",
    # )

    # if wandb_sync:
    #     notify_ending(f"Run {id_string} finished.")

    # wandb.config.update(env_config_dict)
    # wandb.config.update(graph_config_dict)
    # wandb.config.update(model_config)
    # wandb.config.update(config)
    # wandb.config.update(stop)


if __name__ == "__main__":
    main(**parse_args())
