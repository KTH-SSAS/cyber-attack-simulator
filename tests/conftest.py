import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest

from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.utils.config import EnvConfig

from maturin import import_hook

# install the import hook with default settings
import_hook.install()

from attack_simulator.rusty_sim import RustAttackSimulator


@pytest.fixture(name="config_yaml", scope="session")
def fixture_config_filename() -> Path:
    return Path("config/test_config.yaml")


@pytest.fixture(scope="session")
def test_defense_steps(env_config: EnvConfig) -> List[str]:
    with open(env_config.graph_filename, "r", encoding="utf8") as f:
        text = f.read()

    defense_steps = re.findall(r"id: (\w\.(defend))", text)

    assert len(defense_steps) == 2

    return [d[0] for d in defense_steps]


@pytest.fixture(scope="session")
def test_attack_steps(env_config: EnvConfig) -> List[str]:
    with open(env_config.graph_filename, "r", encoding="utf8") as f:
        text = f.read()

    attack_steps = re.findall(r"id: (\w\.(attack|capture))", text)

    assert len(attack_steps) == 6

    return [t[0] for t in attack_steps]

@pytest.fixture(scope="session", name="env_config")
def fixture_env_config(config_yaml: Path) -> EnvConfig:
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)

    absolute_graph_pathname = Path(config.graph_filename).absolute()
    config = dataclasses.replace(
        config, graph_filename=str(absolute_graph_pathname)
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

@pytest.fixture(scope="session", name="simulator")
def fixture_simulator(request):
    config_yaml = Path("config/test_config.yaml")
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)
    return RustAttackSimulator(
        json.dumps(dataclasses.asdict(config.sim_config)), config.graph_filename
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
