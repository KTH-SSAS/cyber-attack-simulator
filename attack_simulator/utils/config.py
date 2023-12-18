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
class EnvConfig(Config):
    """Config class for RL environment."""

    sim_config: SimulatorConfig
    graph_name: str
    vocab_filename: str = None
    attacker_only: bool = False
    undirected_defenses: bool = False

    @classmethod
    def from_dict(cls, dictionary: dict) -> EnvConfig:
        sim_keys = [key for key in dictionary if key.startswith("sim_")]
        sim_config = {key[len("sim_") :]: dictionary.pop(key) for key in sim_keys}
        dictionary["sim_config"] = SimulatorConfig(**sim_config)
        return cls(**dictionary)

    @classmethod
    def from_yaml(cls, filename: str) -> EnvConfig:
        """Load configuration data from YAML file."""
        with open(filename, encoding="utf8") as f:
            dictionary: Dict = yaml.safe_load(f)
            return cls.from_dict(dictionary)


@dataclass(frozen=True)
class SimulatorConfig(Config):
    """Config class for attack simulator."""

    false_negative_rate: float = 0.0
    false_positive_rate: float = 0.0
    randomize_ttc: bool = False  # Randomize time to compromise values
    log: bool = False  # Log simulator output to file
    show_false: bool = False  # Show false positives in render
    strict: bool = False  # Restrict actions to valid ones

    @classmethod
    def from_yaml(cls, filename: str) -> SimulatorConfig:
        """Load configuration data from YAML file."""
        with open(filename, encoding="utf8") as f:
            dictionary = yaml.safe_load(f)
            dictionary = dictionary["sim_config"]
            return cls(**dictionary)
