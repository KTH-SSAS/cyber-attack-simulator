import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from attack_simulator import examplemanager

from attack_simulator.env.env import AttackSimulationEnv
from attack_simulator.utils.config import EnvConfig

from maturin import import_hook

# install the import hook with default settings
import_hook.install()

from attack_simulator.rusty_sim import RustAttackSimulator


@pytest.fixture(name="config_yaml", scope="session")
def fixture_config_filename() -> Path:
    return Path("config/test_config.yaml")

@pytest.fixture(scope="session", name="env_config")
def fixture_env_config(config_yaml: Path) -> EnvConfig:
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)
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
    graph_filename = examplemanager.get_paths_to_graphs()[config.graph_name]
    return RustAttackSimulator(
        json.dumps(dataclasses.asdict(config.sim_config)), graph_filename
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
