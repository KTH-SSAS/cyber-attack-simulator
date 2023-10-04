import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.mal.graph import AttackGraph
from attack_simulator.mal.sim import AttackSimulator
from attack_simulator.utils.config import EnvConfig, GraphConfig

from maturin import import_hook

# install the import hook with default settings
import_hook.install()

from attack_simulator.rusty_sim import RustAttackSimulator


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
    absolute_graph_pathname = Path(config.filename).absolute()
    config = dataclasses.replace(config, filename=str(absolute_graph_pathname))
    return config


@pytest.fixture(scope="session", name="env_config")
def fixture_env_config(config_yaml: Path) -> EnvConfig:
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)

    absolute_graph_pathname = Path(config.graph_config.filename).absolute()
    config = dataclasses.replace(
        config,
        graph_config=dataclasses.replace(
            config.graph_config, filename=str(absolute_graph_pathname)
        ),
    )

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


@pytest.fixture(
    name="simulator",
    params=[
        # pytest.param(AttackSimulator, id="python"),
        pytest.param(RustAttackSimulator, id="rust"),
    ],
)
def fixture_simulator(request) -> AttackSimulator:
    config_yaml = Path("config/test_config.yaml")
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)
    return RustAttackSimulator(
        json.dumps(dataclasses.asdict(config.sim_config)), config.graph_config.filename
    )


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
