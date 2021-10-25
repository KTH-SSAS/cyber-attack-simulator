import argparse
import os
from ray import tune
import yaml

from attack_simulator.agents import ATTACKERS, DEFENDERS
from attack_simulator.config import config_from_dicts
from dataclasses import asdict
from attack_simulator.custom_callback import AttackSimCallback
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.graph import SIZES
from ray.tune.integration.wandb import WandbLoggerCallback
import ray
import wandb


def dict2choices(d):
    choices = list(d.keys())
    choices_help = '", "'.join(choices[:-1]) + f'" or "{choices[-1]}'
    return choices, choices_help


def parse_args():
    sizes, sizes_help = dict2choices(SIZES)
    defenders, defenders_help = dict2choices(DEFENDERS)
    attackers, attackers_help = dict2choices(ATTACKERS)

    parser = argparse.ArgumentParser(
        description="Reinforcement learning of a computer network defender, using RLlib"
    )

    parser.add_argument("-g", "--graph", action="store_true", help="Generate a GraphViz .dot file.")

    parser.add_argument("--graph-config", type=str)

    parser.add_argument(
        "-D",
        "--defender",
        metavar="DEFENDER",
        choices=defenders,
        type=str,
        default=defenders[-1],
        help=f'Select defender. Choices are "{defenders_help}".  Default is "{defenders[-1]}"',
    )

    parser.add_argument(
        "-A",
        "--attacker",
        metavar="ATTACKER",
        choices=attackers,
        type=str,
        default=attackers[-1],
        help=f'Select attacker. Choices are "{attackers_help}".  Default is "{attackers[-1]}"',
    )

    parser.add_argument(
        "-s",
        "--graph_size",
        metavar="SIZE",
        choices=sizes,
        type=str,
        default=sizes[-1],
        help=f'Run simulations on a "{sizes_help}" attack graph. Default is "{sizes[-1]}".',
    )

    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for simulation. Default is None, which falls back to OS entropy.",
    )

    parser.add_argument(
        "-R",
        "--same_seed",
        action="store_true",
        help="Use the same seed for BOTH training AND evaluation. Defaults to `False`",
    )

    parser.add_argument(
        "-N",
        "--rollouts",
        type=int,
        default=100,
        help="Number of simulations to run after training, for evaluation.",
    )

    parser.add_argument(
        "--stop-iters", type=int, default=800, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=1000000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--stop-reward", type=float, default=20000.0, help="Reward at which we stop training."
    )

    parser.add_argument("--eval-interval", default=50)

    parser.add_argument(
        "--render", action="store_true", help="Render an animation of the evaluation."
    )

    parser.add_argument("-L", "--lr", help="Optimizer (Adam) learning rate.", default=1e-2)

    parser.add_argument("-C", "--cuda", action="store_true", help="Use CUDA acceleration.")

    parser.add_argument("--wandb-sync", action="store_true", help="Sync run with wandb cloud.")

    return parser.parse_args()


def main(args):

    dashboard_host = "0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1"
    ray.init(dashboard_host=dashboard_host)

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

    with open("config/default_env_config.yaml") as f:
        env_config_dict = yaml.safe_load(f)

    with open("config/default_graph_config.yaml") as f:
        graph_config_dict = yaml.safe_load(f)

    env_config, _ = config_from_dicts(graph_config_dict, env_config_dict)

    model_config = {"use_lstm": True, "lstm_cell_size": 256}

    config = {
        "framework": "torch",
        "env": AttackSimulationEnv,
        "model": model_config,
        "env_config": asdict(env_config),
        # The number of iterations between renderings
        "evaluation_interval": args.eval_interval,
        "evaluation_num_episodes": 1,
        #(setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        "evaluation_num_workers": 1,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            "render_env": args.render,
            # workaround for a bug in RLLib (https://github.com/ray-project/ray/issues/17921)
            "replay_sequence_length": -1,
        },
        "num_workers": 1,
        "callbacks": AttackSimCallback,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    analysis = tune.run(
        "PPO",
        config=config,
        stop=stop,
        callbacks=callbacks,
        checkpoint_at_end=True,
        metric="episode_reward_mean",
        mode="max",
        checkpoint_freq=1,
        checkpoint_score_attr='episode_reward_mean'
    )

    pass
    # wandb.config.update(env_config_dict)
    # wandb.config.update(graph_config_dict)
    # wandb.config.update(model_config)
    # wandb.config.update(config)
    # wandb.config.update(stop)


if __name__ == "__main__":
    args = parse_args()
    main(args)
