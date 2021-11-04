from dataclasses import asdict, dataclass
from typing import Optional

from .agents import DEFENDERS
from .env import AttackSimulationEnv
from .graph import AttackGraph


@dataclass
class Config:
    def to_dict(self):
        dictionary = asdict(self)
        if dictionary.get("attack_graph"):  # Since the attack graph is an object, it is excluded.
            del dictionary["attack_graph"]
        return dictionary


@dataclass
class GraphConfig(Config):
    low_flag_reward: int
    medium_flag_reward: int
    high_flag_reward: int
    easy_ttc: int
    hard_ttc: int
    graph_size: str


@dataclass
class EnvConfig(Config):
    attack_graph: AttackGraph
    attacker: str
    true_positive: float
    false_positive: float
    save_graphs: bool
    save_logs: bool


@dataclass
class AgentConfig(Config):
    agent_type: str
    random_seed: Optional[int]
    input_dim: int
    hidden_dim: int
    num_actions: int
    learning_rate: float
    use_cuda: bool
    attack_graph: AttackGraph


def create_graph(graph_config: GraphConfig, **kwargs):
    return AttackGraph(dict(asdict(graph_config), **kwargs))


def create_env(env_config: EnvConfig, **kwargs):
    return AttackSimulationEnv(dict(asdict(env_config), **kwargs))


def create_agent(agent_config: AgentConfig, **kwargs):
    config = dict(asdict(agent_config), **kwargs)
    return DEFENDERS[config["agent_type"]](config)


def config_from_dicts(graph_config_dict, env_config_dict):
    graph_config = GraphConfig(**graph_config_dict)

    attack_graph = create_graph(graph_config)

    env_config = EnvConfig(attack_graph=attack_graph, **env_config_dict)

    return env_config, graph_config


def make_configs(parsed_args):
    """Create base configurations from parsed arguments"""

    graph_config = GraphConfig(
        low_flag_reward=parsed_args.low_flag_reward,
        medium_flag_reward=parsed_args.medium_flag_reward,
        high_flag_reward=parsed_args.high_flag_reward,
        easy_ttc=parsed_args.easy_ttc,
        hard_ttc=parsed_args.hard_ttc,
        graph_size=parsed_args.graph_size,
    )
    attack_graph = create_graph(graph_config)

    env_config = EnvConfig(
        attack_graph=attack_graph,
        attacker=parsed_args.attacker,
        true_positive=parsed_args.true_positive_training,
        false_positive=parsed_args.false_positive_training,
        save_graphs=parsed_args.graph,
        save_logs=parsed_args.save_logs
    )
    # need a dummy env to get dimensions for action/observation spaces
    dummy_env = create_env(env_config)

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
