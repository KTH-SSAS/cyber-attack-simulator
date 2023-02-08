import dataclasses
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from attack_simulator.config import EnvConfig, GraphConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.graph import AttackGraph
from attack_simulator.sim import AttackSimulator


@pytest.fixture(name="config_yaml", scope="session")
def fixture_config_filename() -> Path:
    return Path("config/test_config.yaml")


@pytest.fixture(scope="session")
def test_defense_steps(graph_config: GraphConfig) -> List[str]:
    with open(graph_config.filename, "r", encoding="utf8") as f:
        text = f.read()

    defense_steps = re.findall(r"id: (\w\.(defend))", text)

    assert len(defense_steps) == 2

    return [d[0] for d in defense_steps]


@pytest.fixture(scope="session")
def test_attack_steps(graph_config: GraphConfig) -> List[str]:
    with open(graph_config.filename, "r", encoding="utf8") as f:
        text = f.read()

    attack_steps = re.findall(r"id: (\w\.(attack|capture))", text)

    assert len(attack_steps) == 6

    return [t[0] for t in attack_steps]


# @pytest.fixture(scope="session", name="test_assets")
# def fixture_test_assets() -> Dict[str, Any]:
#     return ["a", "b", "c", "d"]


@pytest.fixture(scope="session", name="graph_config")
def fixture_graph_config(config_yaml: Path) -> GraphConfig:
    config: GraphConfig = GraphConfig.from_yaml(config_yaml)
    config = dataclasses.replace(
        config,
    )
    return config


@pytest.fixture(name="attack_graph")
def fixture_graph(graph_config: GraphConfig) -> AttackGraph:
    return AttackGraph(graph_config)


@pytest.fixture(scope="session", name="env_config")
def fixture_env_config(config_yaml: Path) -> EnvConfig:
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)
    dataclasses.replace(config, attacker="well-informed")
    return config


@pytest.fixture(scope="session", name="env")
def fixture_env(env_config: EnvConfig) -> AttackSimulationEnv:
    return AttackSimulationEnv(env_config)


@pytest.fixture(name="simulator")
def fixture_simulator(env_config: EnvConfig, attack_graph: AttackGraph) -> AttackSimulator:
    return AttackSimulator(env_config.sim_config, attack_graph)


@pytest.fixture(scope="session")
def rllib_config(env_config: EnvConfig) -> Dict[str, Any]:
    model_config = {"use_lstm": True, "lstm_cell_size": 256}

    config = {
        "seed": env_config.seed,
        "framework": "torch",
        "env": AttackSimulationEnv,
        "env_config": dataclasses.asdict(env_config),
        "num_workers": 0,
        "model": model_config,
    }

    return config
