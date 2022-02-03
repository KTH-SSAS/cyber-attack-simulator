import dataclasses

import pytest
import yaml

from attack_simulator.config import EnvConfig, GraphConfig
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
    "graph_config": {
        "graph_size": "full",
        "easy_ttc": TTC_LOW,
        "hard_ttc": TTC_HIGH,
        "high_flag_reward": REWARD_HIGH,
        "medium_flag_reward": REWARD_MEDIUM,
        "low_flag_reward": REWARD_LOW,
        "root": "a.x",
        "prune": [],
        "unmalleable_assets": {"internet", "office_network", "hidden_network"},
        "seed": 42,
    },
}

TEST_GRAPH_YAML = """\
---
a.x:
  children:
  - b.x
  conditions:
  - a
  name: x
b.flag.capture:
  conditions:
  - b
  - b.flag
  name: capture
  reward: LOW_FLAG_REWARD
b.u.y:
  children:
  - c.x
  conditions:
  - b
  - b.u
  name: y
  ttc: EASY_TTC
b.v.flag.capture:
  conditions:
  - b
  - b.v
  - b.v.flag
  name: capture
  reward: MEDIUM_FLAG_REWARD
b.v.y:
  children:
  - b.v.flag.capture
  - c.x
  conditions:
  - b
  - b.v
  name: y
  ttc: HARD_TTC
b.x:
  children:
  - b.flag.capture
  - b.u.y
  - b.v.y
  conditions:
  - b
  name: x
  ttc: EASY_TTC
c.u.flag.capture:
  conditions:
  - c
  - c.u
  - c.u.flag
  name: capture
  reward: HIGH_FLAG_REWARD
c.u.x:
  children:
  - c.u.flag.capture
  conditions:
  - c
  - c.u
  name: x
  ttc: HARD_TTC
c.x:
  children:
  - c.u.x
  conditions:
  - c
  name: x
  step_type: and
"""

TEST_SERVICES = ["a", "b", "b.u", "b.v", "c", "c.u"]

TEST_ATTACK_STEPS = {
    "a.x": AttackStep(name="x", conditions=["a"], children=["b.x"]),
    "b.x": AttackStep(
        name="x",
        conditions=["b"],
        ttc=TTC_LOW,
        children=[
            "b.flag.capture",
            "b.u.y",
            "b.v.y",
        ],
        parents=["a.x"],
    ),
    "b.flag.capture": AttackStep(
        name="capture",
        conditions=["b", "b.flag"],
        reward=REWARD_LOW,
        parents=["b.x"],
    ),
    "b.u.y": AttackStep(
        name="y",
        conditions=["b", "b.u"],
        ttc=TTC_LOW,
        children=["c.x"],
        parents=["b.x"],
    ),
    "b.v.y": AttackStep(
        name="y",
        conditions=["b", "b.v"],
        ttc=TTC_HIGH,
        children=["b.v.flag.capture", "c.x"],
        parents=["b.x"],
    ),
    "b.v.flag.capture": AttackStep(
        name="capture",
        conditions=["b", "b.v", "b.v.flag"],
        reward=REWARD_MEDIUM,
        parents=["b.v.y"],
    ),
    "c.x": AttackStep(
        name="x",
        step_type="and",
        conditions=["c"],
        children=["c.u.x"],
        parents=["b.u.y", "b.v.y"],
    ),
    "c.u.x": AttackStep(
        name="x",
        conditions=["c", "c.u"],
        ttc=TTC_HIGH,
        children=["c.u.flag.capture"],
        parents=["c.x"],
    ),
    "c.u.flag.capture": AttackStep(
        name="capture",
        conditions=["c", "c.u", "c.u.flag"],
        reward=REWARD_HIGH,
        parents=["c.u.x"],
    ),
}

TEST_DOT = """
digraph G {
"1 :: a.x, 1" -> "6 :: b.x, 10";
"3 :: b.u.y, 10" -> "9 :: c.x, 1";
"5 :: b.v.y, 100" -> "4 :: b.v.flag.capture, 1";
"5 :: b.v.y, 100" -> "9 :: c.x, 1";
"6 :: b.x, 10" -> "2 :: b.flag.capture, 1";
"6 :: b.x, 10" -> "3 :: b.u.y, 10";
"6 :: b.x, 10" -> "5 :: b.v.y, 100";
"8 :: c.u.x, 100" -> "7 :: c.u.flag.capture, 1";
"9 :: c.x, 1" -> "8 :: c.u.x, 100";
}
"""


@pytest.fixture(scope="session")
def test_services():
    return TEST_SERVICES


@pytest.fixture(scope="session")
def test_attack_steps():
    return TEST_ATTACK_STEPS


@pytest.fixture(scope="session")
def test_graph_dot():
    return TEST_DOT.strip()


@pytest.fixture(scope="session", name="graph_yaml")
def fixture_graph_yaml(tmpdir_factory):
    graph_yaml = tmpdir_factory.mktemp("test").join("graph.yaml")
    graph_yaml.write(TEST_GRAPH_YAML)
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
