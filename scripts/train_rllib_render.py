#!/usr/bin/env python3

# IMPORTANT NOTE:
# ---------------
# A recent bug in openAI gym prevents RLlib's "record_env" option
# from recording videos properly. Instead, the produced mp4 files
# have a size of 1kb and are corrupted.
# A simple fix for this is described here:
# https://github.com/openai/gym/issues/1925

import argparse

import ray
from ray import tune

from attack_simulator.env import AttackSimulationEnv

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO", help="The RLlib-registered algorithm to use.")
parser.add_argument("--stop-iters", type=int, default=800, help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps", type=int, default=1000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=20000.0, help="Reward at which we stop training."
)


if __name__ == "__main__":
    # Note: Recording and rendering in this example
    # should work for both local_mode=True|False.
    ray.init(num_cpus=4)
    args = parser.parse_args()

    config = {
        "env": AttackSimulationEnv,
        # The number of iterations between renderings
        "evaluation_interval": 50,
        # Run evaluation on (at least) two episodes
        "evaluation_num_episodes": 1,
        # ... using one evaluation worker (setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        "evaluation_num_workers": 1,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            "render_env": True,
        },
        "num_workers": 1,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run("PPO", config=config, stop=stop)
