import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from attack_simulator import examplemanager

from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.utils.config import EnvConfig
from attack_simulator.rusty_sim import RustAttackSimulator
from maturin import import_hook

# install the import hook with default settings
import_hook.install()

from attack_simulator.rusty_sim import RustAttackSimulator


@pytest.fixture(name="config_yaml", scope="session")
def fixture_config_filename() -> Path:
    return Path("config/test_config.yaml")

@pytest.fixture(scope="session", name="env_config")
def fixture_env_config(config_yaml: Path) -> EnvConfig:
    conf: Dict[str, Any] = {
        "sim_false_positive_rate": 0.2,
        "sim_false_negative_rate": 0.2,
        "graph_name": "test_graph"
    }
    config: EnvConfig = EnvConfig.from_dict(conf)
    return config


@pytest.fixture(
    scope="session",
    name="env",
    params=[
        # pytest.param("python", id="python"),
        pytest.param("rust", id="rust"),
    ],
)
def fixture_env(request) -> AttackSimulationEnv:
    env_config = request.getfixturevalue("env_config")
    return AttackSimulationEnv(env_config)


@pytest.fixture(scope="session", name="simulator")
def fixture_simulator(request) -> RustAttackSimulator:
    conf: Dict[str, Any] = {
        "sim_false_positive_rate": 0.2,
        "sim_false_negative_rate": 0.2,
        "graph_name": "test_graph"
    }
    config: EnvConfig = EnvConfig.from_dict(conf)
    graph_filename: str = examplemanager.get_paths_to_graphs()[config.graph_name]
    return RustAttackSimulator(
        json.dumps(dataclasses.asdict(config.sim_config)), graph_filename
    )