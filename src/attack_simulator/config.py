from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Base config class."""

    def replace(self, **kwargs: Any) -> Config:
        """Wrapper function for dataclasses.replace."""
        return dataclasses.replace(self, **kwargs)


@dataclass(frozen=True)
class GraphConfig(Config):
    """Config class for attack graph."""

    low_flag_reward: int
    medium_flag_reward: int
    high_flag_reward: int
    easy_ttc: int
    hard_ttc: int
    filename: str
    root: str
    prune: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, filename: str) -> GraphConfig:
        """Load configuration data from YAML file."""
        with open(filename, encoding="utf8") as f:
            dictionary = yaml.safe_load(f)
            dictionary = dictionary["graph_config"]
            return cls(**dictionary)


@dataclass(frozen=True)
class EnvConfig(Config):
    """Config class for RL environment."""

    graph_config: GraphConfig
    attacker: str
    false_negative: float
    false_positive: float
    save_graphs: bool
    save_logs: bool
    attack_start_time: int
    reward_mode: str
    seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, filename: str) -> EnvConfig:
        """Load configuration data from YAML file."""
        with open(filename, encoding="utf8") as f:
            dictionary = yaml.safe_load(f)
            dictionary["graph_config"] = GraphConfig(**dictionary["graph_config"])
            return cls(**dictionary)


@dataclass(frozen=True)
class AgentConfig(Config):
    """Config class for RL agents."""

    agent_type: str
    seed: Optional[int]
    input_dim: int
    hidden_dim: int
    num_actions: int
    learning_rate: float
    use_cuda: bool
