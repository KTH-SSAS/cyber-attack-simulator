from .config import AgentConfig, EnvConfig, GraphConfig
from .env import AttackSimulationEnv
from .graph import AttackGraph


def config_from_dicts(env_config_dict):
    # graph_config = GraphConfig(**graph_config_dict)
    env_config = EnvConfig(**env_config_dict)

    return env_config


def config_from_args(parsed_args):
    """Create base configurations from parsed arguments"""

    graph_config = GraphConfig(
        low_flag_reward=parsed_args.low_flag_reward,
        medium_flag_reward=parsed_args.medium_flag_reward,
        high_flag_reward=parsed_args.high_flag_reward,
        easy_ttc=parsed_args.easy_ttc,
        hard_ttc=parsed_args.hard_ttc,
        graph_size=parsed_args.graph_size,
    )
    attack_graph = AttackGraph(graph_config)

    env_config = EnvConfig(
        attacker=parsed_args.attacker,
        true_positive=parsed_args.true_positive_training,
        false_positive=parsed_args.false_positive_training,
        save_graphs=parsed_args.graph,
        save_logs=parsed_args.save_logs,
    )
    # need a dummy env to get dimensions for action/observation spaces
    dummy_env = AttackSimulationEnv(env_config)

    agent_config = AgentConfig(
        agent_type=parsed_args.defender,
        random_seed=parsed_args.random_seed,
        input_dim=len(dummy_env.observation_space.spaces),
        hidden_dim=parsed_args.hidden_width,
        num_actions=dummy_env.action_space.n,
        learning_rate=parsed_args.lr,
        use_cuda=parsed_args.cuda,
        attack_graph=attack_graph,
        # disable_probability=0.2,
        # gamma=1,
    )

    del dummy_env

    return agent_config, env_config, graph_config
