import os
import socket
from datetime import datetime

import ray
import ray.train.torch

from ray.tune.stopper import MaximumIterationStopper
from ray.rllib.algorithms.impala import ImpalaTorchPolicy, ImpalaConfig, Impala
from ray import air, tune
import attack_simulator

from rllib_common import get_config, setup_wandb

def main(
    config_file: str,
    **kwargs,
) -> None:
    """Main function for running the RLlib experiment."""

    dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"
    ray.init(dashboard_host=dashboard_host)


    callbacks = []
    wandb = setup_wandb(kwargs['wandb_sync'])
    if wandb is not None:
        callbacks.append(wandb)

    config = get_config(config_file, gpu_count=0)

    metric = "info/learner/defender/learner_stats/vf_loss"
    max_iteration = 100
    perturbation_interval = 10
    num_samples = 1

    restore = False

    tuner = (
        tune.Tuner(
            Impala,
            tune_config=tune.TuneConfig(
                reuse_actors=False,
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
        else tune.Tuner.restore(path="~/ray_results/Impala_2023-09-28_10-13-00/")
    )

    tuner.fit()

if __name__ == "__main__":
     main("config/maze_env_config.yaml", wandb_sync=True)
