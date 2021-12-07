import argparse
import dataclasses
import os
from dataclasses import asdict

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback

from attack_simulator.config import EnvConfig
from attack_simulator.custom_callback import AttackSimCallback
from attack_simulator.env import AttackSimulationEnv


def dict2choices(d):
    choices = list(d.keys())
    choices_help = '", "'.join(choices[:-1]) + f'" or "{choices[-1]}'
    return choices, choices_help


def parse_args():

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

    parser.add_argument("--local", action="store_true", help="Enable ray local mode for debugger.")

    return parser.parse_args()


def main(args):

    dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"

    if args.local:  # TODO might want to set other options if we know the program is running locally
        local = True
    else:
        local = False

    ray.init(dashboard_host=dashboard_host, local_mode=local)

    callbacks = []

    os.environ["WANDB_MODE"] = "online" if args.wandb_sync else "offline"

    wandb_api_file_exists = os.path.exists("./wandb_api_key")
    if wandb_api_file_exists or os.environ.get("WANDB_API_KEY") is not None:

        api_key_file = "./wandb_api_key" if wandb_api_file_exists else None

        callbacks.append(
            WandbLoggerCallback(
                project="rl_attack_sim",
                group="single",
                api_key_file=api_key_file,
                log_config=False,
                entity="sentience",
            )
        )

    env_config = EnvConfig.from_yaml(args.config_file)

    env_config = dataclasses.replace(env_config, save_graphs=args.graph)

    model_config = {"use_lstm": True, "lstm_cell_size": 256}

    gpu_count = args.gpu_count
    batch_size = 4000
    num_workers = 5
    env_per_worker = 5
    # fragment_length = 200

    config = {
        "seed": env_config.seed,
        "framework": "torch",
        "env": AttackSimulationEnv,
        "num_gpus": gpu_count,
        "train_batch_size": batch_size * env_per_worker,
        "num_envs_per_worker": env_per_worker,
        "model": model_config,
        "env_config": asdict(env_config),
        "batch_mode": "complete_episodes",
        "sgd_minibatch_size": 256,
        # The number of iterations between renderings
        # "evaluation_interval": args.eval_interval,
        # "evaluation_num_episodes": 1,
        # (setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        # "evaluation_num_workers": 1,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        # "evaluation_config": {
        # Render the env while evaluating.
        # Note that this will always only render the 1st RolloutWorker's
        # env and only the 1st sub-env in a vectorized env.
        #    "render_env": args.render,
        # workaround for a bug in RLLib (https://github.com/ray-project/ray/issues/17921)
        # "replay_sequence_length": -1,
        # },
        "num_workers": num_workers,
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

    tune.run(
        "PPO",
        config=config,
        stop=stop,
        callbacks=callbacks,
        checkpoint_at_end=True,
        metric="episode_reward_mean",
        mode="max",
        keep_checkpoints_num=5,
        checkpoint_freq=1,
        checkpoint_score_attr="episode_reward_mean",
        restore=args.checkpoint_path,
    )

    # wandb.config.update(env_config_dict)
    # wandb.config.update(graph_config_dict)
    # wandb.config.update(model_config)
    # wandb.config.update(config)
    # wandb.config.update(stop)


if __name__ == "__main__":
    args = parse_args()
    main(args)
