import dataclasses

import pytest
import yaml

from attack_simulator.config import EnvConfig, GraphConfig
from attack_simulator.constant import AND, DEFENSE
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.graph import AttackGraph, AttackStep
from attack_simulator.sim import AttackSimulator

REWARD_HIGH = 1000
REWARD_MEDIUM = 100
REWARD_LOW = 10

TTC_HIGH = 100
TTC_LOW = 10

TEST_ENV_CONFIG_YAML = {
    "attacker": "random",
    "false_positive": 0.0,
    "save_graphs": False,
    "save_logs": False,
    "false_negative": 0.0,
    "attack_start_time": 0,
    "seed": 42,
    "reward_mode": "simple",
    "run_id": "test",
    "graph_config": {
        "easy_ttc": TTC_LOW,
        "hard_ttc": TTC_HIGH,
        "high_flag_reward": REWARD_HIGH,
        "medium_flag_reward": REWARD_MEDIUM,
        "low_flag_reward": REWARD_LOW,
        "root": "a.attack",
    },
}

"""Simple model for testing. C and B are children of A. B has to be compromised before D."""
TEST_GRAPH = {
    "attack_graph": [
        {
            "id": "a.attack",
            "children": ["b.attack", "c.attack"],
            "asset": "a",
            "name": "attack",
            "ttc": "EASY_TTC",
        },
        {
            "id": "b.attack",
            "ttc": "EASY_TTC",
            "children": ["b.capture", "d.attack"],
            "asset": "b",
            "name": "attack",
        },
        {
            "id": "b.capture",
            "asset": "b",
            "name": "capture",
            "ttc": "EASY_TTC",
        },
        {
            "id": "c.attack",
            "children": ["d.attack"],
            "asset": "c",
            "name": "attack",
            "ttc": "EASY_TTC",
        },
        {
            "id": "d.attack",
            "children": ["d.capture"],
            "asset": "d",
            "name": "attack",
            "step_type": AND,
            "ttc": "EASY_TTC",
        },
        {
            "id": "d.capture",
            "asset": "d",
            "name": "capture",
            "ttc": "HARD_TTC",
        },
        {
            "id": "c.defend",
            "step_type": DEFENSE,
            "name": "defend",
            "children": ["c.attack"],
            "asset": "c",
            "reward": 2,
        },
        {
            "id": "b.defend",
            "name": "defend",
            "step_type": DEFENSE,
            "children": ["b.attack", "b.capture"],
            "asset": "b",
            "reward": 1,
        },
    ],
    "instance_model": [
        {"id": "a", "dependents": ["b"]},
        {"id": "b", "dependents": []},
        {"id": "c", "dependents": ["d"]},
        {"id": "d", "dependents": []},
    ],
    "flags": {"b.capture": "LOW_FLAG_REWARD", "d.capture": "HIGH_FLAG_REWARD"}
}

TEST_ASSETS = ["a", "b", "c", "d"]

TEST_ATTACK_STEPS = {
    "a.attack": AttackStep(
        id="a.attack", children=["b.attack", "c.attack"], asset="a", ttc=TTC_LOW
    ),
    "b.attack": AttackStep(
        id="b.attack",
        name="attack",
        ttc=TTC_LOW,
        children=["b.capture", "d.attack"],
        parents=["a.attack"],
        asset="b",
    ),
    "b.capture": AttackStep(
        id="b.capture",
        parents=["b.attack"],
        name="capture",
        asset="b",
        ttc=TTC_LOW,
    ),
    "c.attack": AttackStep(
        id="c.attack", ttc=TTC_LOW, children=["d.attack"], parents=["a.attack"], asset="c"
    ),
    "d.attack": AttackStep(
        id="d.attack",
        ttc=TTC_LOW,
        children=["d.capture"],
        parents=["b.attack", "c.attack"],
        step_type=AND,
        asset="d",
    ),
    "d.capture": AttackStep(
        id="d.capture",
        ttc=TTC_HIGH,
        parents=["d.attack"],
        asset="d",
        name="capture",
    ),
}


@pytest.fixture(scope="session")
def test_services():
    return TEST_ASSETS


@pytest.fixture(scope="session")
def test_defense_steps():
    return ["b.defend", "c.defend"]


@pytest.fixture(scope="session")
def test_attack_steps():
    return TEST_ATTACK_STEPS


@pytest.fixture(scope="session", name="graph_yaml")
def fixture_graph_yaml(tmpdir_factory):
    folder = tmpdir_factory.mktemp("test")
    graph_yaml = folder.join("test.yaml")
    graph_yaml.write(yaml.dump(TEST_GRAPH))
    return graph_yaml


@pytest.fixture(scope="session", name="config_yaml")
def fixture_config_yaml(tmpdir_factory, graph_yaml):
    config = tmpdir_factory.mktemp("graph_config").join("config.yaml")
    TEST_ENV_CONFIG_YAML["graph_config"]["filename"] = str(graph_yaml)
    to_write = yaml.dump(TEST_ENV_CONFIG_YAML)

    config.write(to_write)
    return config


@pytest.fixture(scope="session", name="graph_config")
def fixture_graph_config(config_yaml):

    config: GraphConfig = GraphConfig.from_yaml(config_yaml)
    config = dataclasses.replace(
        config,
    )

    return config


@pytest.fixture(scope="session", name="attack_graph")
def fixture_graph(graph_config):
    return AttackGraph(graph_config)


@pytest.fixture(scope="session", name="env_config")
def fixture_env_config(config_yaml):
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)
    dataclasses.replace(config, attacker="well-informed")
    return config


@pytest.fixture(scope="session", name="env")
def fixture_env(env_config):
    return AttackSimulationEnv(env_config)


@pytest.fixture(name="simulator")
def fixture_simulator(env):
    return AttackSimulator(env.config, env.rng)


@pytest.fixture(scope="session")
def rllib_config(env_config):
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
