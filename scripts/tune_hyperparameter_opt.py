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
from ray.rllib.algorithms.impala import ImpalaTorchPolicy, ImpalaConfig, Impala
from ray.rllib.algorithms.ppo import PPO

import attack_simulator

from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.rllib.attackers_policies import BreadthFirstPolicy, DepthFirstPolicy
from attack_simulator.rllib.custom_callback import AttackSimCallback
from attack_simulator.utils.config import EnvConfig

from rllib_common import get_config, setup_wandb

def main(
    config_file: str,
    **kwargs,
) -> None:
    """Main function for running the RLlib experiment."""

    dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"

    ray.init(dashboard_host=dashboard_host)

    callbacks = []
    wandb = setup_wandb(True)
    if wandb is not None:
        callbacks.append(wandb)

    config = get_config(config_file, gpu_count=0)

    metric = "info/learner/defender/learner_stats/vf_loss"
    mode = "min"
    method = "pbt"
    perturb = 0.25
    max_iteration = 100
    perturbation_interval = 10
    num_samples = 1
    hyperparam_bounds = {
        "lr": [1e-4, 1e-1],
        "vf_clip_param": [1e-3, 1e3],
        "vf_loss_coeff": [0.01, 100],
        "clip_param": [0.0, 10],
        # "entropy_coeff": [0.0, 1.0],
        # "kl_coeff": [0.0, 1.0],
        "lambda": [0.9, 1],
    }

    if method == "pbt2":
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
            Impala,
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
        else tune.Tuner.restore(path="~/ray_results/Impala_2023-09-28_10-58-56/")
    )

    results = tuner.fit()
    result_dfs = [result.metrics_dataframe for result in results if result.metrics_dataframe is not None]
    
    if not result_dfs:
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    for i, df in enumerate(result_dfs):
        plt.plot(df[metric], label=i)
    plt.legend()
    plt.title(f"{metric}")
    plt.xlabel("Training Iterations")
    plt.ylabel(f"{metric}")
    #plt.savefig(f"{metric}.png")
    plt.show()

    plt.figure(figsize=(7, 4))
    for i, df in enumerate(result_dfs):
        plt.plot(df["episode_reward_mean"], label=i)
    plt.legend()
    plt.title("Reward During Training")
    plt.xlabel("Training Iterations")
    plt.ylabel("Reward")
    #plt.savefig("reward.png")
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
