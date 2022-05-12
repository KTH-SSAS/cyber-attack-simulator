from argparse import Namespace
from typing import Tuple

from .config import AgentConfig, EnvConfig, GraphConfig
from .env import AttackSimulationEnv


def config_from_dicts(env_config_dict: dict) -> EnvConfig:
    # graph_config = GraphConfig(**graph_config_dict)
    env_config = EnvConfig(**env_config_dict)

    return env_config


def config_from_args(parsed_args: Namespace) -> Tuple[AgentConfig, EnvConfig, GraphConfig]:
    """Create base configurations from parsed arguments."""

    graph_config = GraphConfig(
        low_flag_reward=parsed_args.low_flag_reward,
        medium_flag_reward=parsed_args.medium_flag_reward,
        high_flag_reward=parsed_args.high_flag_reward,
        easy_ttc=parsed_args.easy_ttc,
        hard_ttc=parsed_args.hard_ttc,
        filename=parsed_args.filename,
        root=parsed_args.root,
    )

    env_config = EnvConfig(
        graph_config=graph_config,
        attacker=parsed_args.attacker,
        false_negative=parsed_args.false_negative_training,
        false_positive=parsed_args.false_positive_training,
        save_graphs=parsed_args.graph,
        save_logs=parsed_args.save_logs,
        reward_mode=parsed_args.reward_mode,
        attack_start_time=parsed_args.start_time,
    )

    # need a dummy env to get dimensions for action/observation spaces
    dummy_env = AttackSimulationEnv(env_config)

    agent_config = AgentConfig(
        agent_type=parsed_args.defender,
        seed=parsed_args.random_seed,
        input_dim=len(dummy_env.observation_space.spaces),
        hidden_dim=parsed_args.hidden_width,
        num_actions=dummy_env.action_space.n,
        learning_rate=parsed_args.lr,
        use_cuda=parsed_args.cuda,
    )

    del dummy_env

    return agent_config, env_config, graph_config
