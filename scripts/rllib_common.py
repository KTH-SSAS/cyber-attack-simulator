import argparse
import dataclasses
import os
import re
import socket
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOTorchPolicy, PPOConfig, PPO
from ray.rllib.algorithms.impala import ImpalaTorchPolicy, ImpalaConfig, Impala

import attack_simulator
from ray.air.integrations.wandb import WandbLoggerCallback
from attack_simulator.constants import AGENT_ATTACKER, AGENT_DEFENDER
from attack_simulator.rllib.attackers_policies import BreadthFirstPolicy, DepthFirstPolicy
from attack_simulator.rllib.custom_callback import AttackSimCallback
from attack_simulator.utils.config import EnvConfig

def setup_wandb(wandb_sync: bool):
    os.environ["WANDB_MODE"] = "online" if wandb_sync else "offline"
    attack_simulator.register_rllib()
    current_time = datetime.now().strftime(r"%m-%d_%H:%M:%S")
    id_string = f"{current_time}@{socket.gethostname()}"
    wandb_api_file_exists = os.path.exists("./wandb_api_key")
    if wandb_api_file_exists or os.environ.get("WANDB_API_KEY") is not None:
        api_key_file = "./wandb_api_key" if wandb_api_file_exists else None
        return WandbLoggerCallback(
                project="rl_attack_sim",
                group=f"Sweep_{id_string}",
                api_key_file=api_key_file,
                log_config=False,
                entity="sentience",
                tags=["train"],
            )
    return None

def get_graph_size(config):
    return int(re.search(r"_(\d+)\.yaml", config["env_config"]["graph_config"]["filename"])[1])


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
    attack_simulator.register_rllib()
    global_seed = 22
    id_string = f"{datetime.now().strftime(r'%m-%d_%H:%M:%S')}@{socket.gethostname()}"
    attacker_policy_class = BreadthFirstPolicy
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

    defender_config = {
        "model": {
            "custom_model": "GNNDefenderModel",
            "fcnet_hiddens": [8],
            "vf_share_layers": True,
            "custom_model_config": {},
        }
    }
    attacker_config = {
    }

    #config_class = PPOConfig
    #policy_class = PPOTorchPolicy
    config_class = ImpalaConfig
    policy_class = ImpalaTorchPolicy

    config = (
        config_class()
        .training(
            #sgd_minibatch_size=128,
            train_batch_size=128,
            #vf_clip_param=1000,
            #gamma=1.0,
            #use_critic=True,
            #use_gae=True,
            #scale_rewards=False,
        )
        .framework("torch")
        .environment("AttackSimulationEnv", env_config=asdict(env_config))
        .debugging(seed=global_seed)
        .resources(num_gpus=num_gpus, num_gpus_per_worker=num_gpus_per_worker)
        #.callbacks(AttackSimCallback)
        .rollouts(
            num_rollout_workers=4,
            num_envs_per_worker=1,
            batch_mode="complete_episodes",
        )
        .multi_agent(
            policies={
                AGENT_DEFENDER: PolicySpec(policy_class=policy_class, config=defender_config),
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