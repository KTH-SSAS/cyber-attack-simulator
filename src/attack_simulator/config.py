from dataclasses import asdict, dataclass, field
from typing import List, Optional, List

import yaml

from .agents import DEFENDERS


@dataclass(frozen=True)
class Config:
    def to_dict(self):
        dictionary = asdict(self)
        if dictionary.get("attack_graph"):  # Since the attack graph is an object, it is excluded.
            del dictionary["attack_graph"]
        return dictionary


@dataclass(frozen=True)
class GraphConfig(Config):
    low_flag_reward: int
    medium_flag_reward: int
    high_flag_reward: int
    easy_ttc: int
    hard_ttc: int
    graph_size: str
    random_seed: int
    filename: str
    root: str
    prune: List[str] = field(default_factory=list)
    unmalleable_assets: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            dictionary = yaml.safe_load(f)
            dictionary = dictionary["graph_config"]
            return cls(**dictionary)


@dataclass(frozen=True)
class EnvConfig(Config):
    graph_config: GraphConfig
    attacker: str
    true_positive: float
    false_positive: float
    save_graphs: bool
    save_logs: bool

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            dictionary = yaml.safe_load(f)
            dictionary["graph_config"] = GraphConfig(**dictionary["graph_config"])
            return cls(**dictionary)


@dataclass(frozen=True)
class AgentConfig(Config):
    agent_type: str
    random_seed: Optional[int]
    input_dim: int
    hidden_dim: int
    num_actions: int
    learning_rate: float
    use_cuda: bool


def create_agent(agent_config: AgentConfig, **kwargs):
    config = dict(asdict(agent_config), **kwargs)
    return DEFENDERS[config["agent_type"]](config)
