from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class Config:
    """Base config class."""

    def replace(self, **kwargs: Any) -> Config:
        """Wrapper function for dataclasses.replace."""
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GraphConfig(Config):
    """Config class for attack graph."""

    filename: str
    vocab_filename: str = None

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
    sim_config: SimulatorConfig
    seed: Optional[int] = None

    @classmethod
    def from_yaml(cls, filename: str) -> EnvConfig:
        """Load configuration data from YAML file."""
        with open(filename, encoding="utf8") as f:
            dictionary = yaml.safe_load(f)
            dictionary["sim_config"] = SimulatorConfig(**dictionary["sim_config"])
            dictionary["graph_config"] = GraphConfig(**dictionary["graph_config"])
            return cls(**dictionary)


@dataclass(frozen=True)
class SimulatorConfig(Config):
    """Config class for attack simulator."""

    false_negative_rate: float
    false_positive_rate: float
    seed: int
    randomize_ttc: bool = False
    log: bool = False

    @classmethod
    def from_yaml(cls, filename: str) -> SimulatorConfig:
        """Load configuration data from YAML file."""
        with open(filename, encoding="utf8") as f:
            dictionary = yaml.safe_load(f)
            dictionary = dictionary["sim_config"]
            return cls(**dictionary)
