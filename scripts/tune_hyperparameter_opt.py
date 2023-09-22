import dataclasses
import os
import pprint
import socket
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import ray
import ray.train.torch
from ray import air, tune
from ray.rllib.policy.policy import PolicySpec
from ray.tune.schedulers.pb2 import PB2
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.stopper import MaximumIterationStopper

import attack_simulator.rllib.gnn_model as gnn_defender
from attack_simulator import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.env.env import AttackSimulationEnv, register_rllib_env
from attack_simulator.rllib.attackers_policies import DepthFirstPolicy
from attack_simulator.rllib.custom_callback import AttackSimCallback
from attack_simulator.rllib.defender_policy import Defender, DefenderConfig, DefenderPolicy
from attack_simulator.utils.config import EnvConfig


def make_graph_filename_absolute(env_config: EnvConfig):
    cwd = os.getcwd()
    absolute_graph_pathname = Path.absolute(Path(cwd, env_config.graph_config.filename))
    env_config = dataclasses.replace(
        env_config,
        graph_config=dataclasses.replace(
            env_config.graph_config, filename=str(absolute_graph_pathname)
        ),
    )
    return env_config


def get_config(config_file, gpu_count):
    env_name = register_rllib_env()
    gnn_defender.register_rllib_model()
    global_seed = 22
    id_string = f"{datetime.now().strftime(r'%m-%d_%H:%M:%S')}@{socket.gethostname()}"
    attacker_policy_class = DepthFirstPolicy
    num_workers = 0

    # Allocate GPU power to workers
    num_gpus = 0.0001 if gpu_count > 0 else 0  # Driver GPU
    num_gpus_per_worker = (
        (gpu_count - num_gpus) / num_workers if num_workers > 0 and gpu_count > 0 else 0
    )

    env_config = EnvConfig.from_yaml(config_file)
    env_config = make_graph_filename_absolute(env_config)
    env_config = dataclasses.replace(env_config, run_id=id_string)
    env_config = dataclasses.replace(env_config, backend="rust")

    dummy_env = AttackSimulationEnv(env_config)
    defender_config = {
        "model": {
            "custom_model": "GNNDefenderModel",
            "fcnet_hiddens": [8],
            "vf_share_layers": True,
            "custom_model_config": {},
        }
    }
    attacker_config = {
        "num_special_actions": dummy_env.num_special_actions,
        "wait_action": dummy_env.sim.wait_action,
        "terminate_action": dummy_env.sim.terminate_action,
    }

    config = (
        DefenderConfig()
        .training(
            sgd_minibatch_size=128,
            train_batch_size=128,
            vf_clip_param=1000,
            gamma=1.0,
            use_critic=True,
            use_gae=True,
            scale_rewards=False,
        )
        .framework("torch")
        .environment(env_name, env_config=asdict(env_config))
        # .debugging(seed=global_seed)
        .resources(num_gpus=num_gpus, num_gpus_per_worker=num_gpus_per_worker)
        .callbacks(AttackSimCallback)
        .rollouts(
            num_rollout_workers=1,
            num_envs_per_worker=1,
            batch_mode="complete_episodes",
        )
        .multi_agent(
            policies={
                AGENT_DEFENDER: PolicySpec(policy_class=DefenderPolicy, config=defender_config),
                AGENT_ATTACKER: PolicySpec(
                    attacker_policy_class,
                    config=attacker_config,
                ),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            policies_to_train=[AGENT_DEFENDER],
        )
    )
    return config


def main(
    config_file: str,
    **kwargs,
) -> None:
    """Main function for running the RLlib experiment."""

    dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"

    ray.init(dashboard_host=dashboard_host)

    callbacks = []

    config = get_config(config_file, gpu_count=0)

    metric = "info/learner/defender/learner_stats/vf_loss"
    mode = "min"
    method = "pbt"
    perturb = 0.25
    max_iteration = 100
    perturbation_interval = 10
    num_samples = 3
    hyperparam_bounds = {
        "lr": [1e-4, 1e-1],
        "vf_clip_param": [1e-3, 1e3],
        "vf_loss_coeff": [0.01, 100],
        "clip_param": [0.0, 10],
        # "entropy_coeff": [0.0, 1.0],
        # "kl_coeff": [0.0, 1.0],
        "lambda": [0.9, 1],
    }

    if method == "pbt":
        scheduler = PB2(
            time_attr="training_iteration",
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            hyperparam_bounds=hyperparam_bounds
            # Specifies the search space for these hyperparams
        )
    else:
        scheduler = None

    bayesopt = BayesOptSearch(metric=metric, mode=mode)
    bayesopt = ConcurrencyLimiter(bayesopt, max_concurrent=1)
    search_alg = bayesopt if method == "bayes" else None
    restore = False

    tuner = (
        tune.Tuner(
            Defender,
            tune_config=tune.TuneConfig(
                reuse_actors=False,
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=num_samples,
            ),
            run_config=air.RunConfig(
                stop=MaximumIterationStopper(max_iteration),
                callbacks=callbacks,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_score_attribute=metric,
                    num_to_keep=5,
                    checkpoint_frequency=perturbation_interval,
                ),
                failure_config=air.FailureConfig(
                    fail_fast=kwargs.get("fail_fast", True),
                ),
                verbose=1,
                # progress_reporter=tune.CLIReporter(max_report_frequency=60),
            ),
            param_space=config,
        )
        if not restore
        else tune.Tuner.restore(path="~/ray_results/Defender_2023-05-05_11-29-26/")
    )

    results = tuner.fit()
    result_dfs = [result.metrics_dataframe for result in results]
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for i, df in enumerate(result_dfs):
        plt.plot(df[metric], label=i)
    plt.legend()
    plt.title(f"{metric}")
    plt.xlabel("Training Iterations")
    plt.ylabel(f"{metric}")
    plt.show()

    plt.figure(figsize=(7, 4))
    for i, df in enumerate(result_dfs):
        plt.plot(df["episode_reward_mean"], label=i)
    plt.legend()
    plt.title("Reward During Training")
    plt.xlabel("Training Iterations")
    plt.ylabel("Reward")
    plt.show()

    best_result = results.get_best_result(metric=metric, mode=mode)
    optimized_params = best_result.config
    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint({k: v for k, v in optimized_params.items() if k in hyperparam_bounds})

    print("\nBest performing trial's final reported metrics:\n")

    metrics_to_print = [
        "episode_reward_mean",
        "episode_reward_max",
        "episode_reward_min",
        "episode_len_mean",
    ]
    pprint.pprint({k: v for k, v in best_result.metrics.items() if k in metrics_to_print})


if __name__ == "__main__":
    main("config/maze_env_config.yaml")
