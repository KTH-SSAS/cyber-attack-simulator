import dataclasses

import pytest
import yaml

from attack_simulator.config import EnvConfig, GraphConfig
from attack_simulator.env import AttackSimulationEnv
from attack_simulator.graph import AttackGraph, AttackStep

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
b.x:
  ttc: EASY_TTC
  children:
    - b.flag.capture
    - b.u.y
    - b.v.y
b.flag.capture:
  reward: LOW_FLAG_REWARD
b.u.y:
  ttc: EASY_TTC
  children:
    - c.x
b.v.y:
  ttc: HARD_TTC
  children:
    - b.v.flag.capture
    - c.x
b.v.flag.capture:
  reward: MEDIUM_FLAG_REWARD
c.x:
  children:
    - c.u.x
  step_type: and
c.u.x:
  ttc: HARD_TTC
  children:
    - c.u.flag.capture
c.u.flag.capture:
  reward: HIGH_FLAG_REWARD
"""

TEST_SERVICES = ["a", "b", "b.u", "b.v", "c", "c.u"]

TEST_ATTACK_STEPS = {
    "a.x": AttackStep(asset="a", name="x", children=["b.x"]),
    "b.x": AttackStep(
        asset="b",
        name="x",
        ttc=TTC_LOW,
        children=[
            "b.flag.capture",
            "b.u.y",
            "b.v.y",
        ],
        parents=["a.x"],
    ),
    "b.flag.capture": AttackStep(
        asset="b",
        flag="flag",
        name="capture",
        reward=REWARD_LOW,
        parents=["b.x"],
    ),
    "b.u.y": AttackStep(
        asset="b",
        service="u",
        name="y",
        ttc=TTC_LOW,
        children=["c.x"],
        parents=["b.x"],
    ),
    "b.v.y": AttackStep(
        asset="b",
        service="v",
        name="y",
        ttc=TTC_HIGH,
        children=["b.v.flag.capture", "c.x"],
        parents=["b.x"],
    ),
    "b.v.flag.capture": AttackStep(
        asset="b",
        service="v",
        flag="flag",
        name="capture",
        reward=REWARD_MEDIUM,
        parents=["b.v.y"],
    ),
    "c.x": AttackStep(
        asset="c",
        name="x",
        step_type="and",
        children=["c.u.x"],
        parents=["b.u.y", "b.v.y"],
    ),
    "c.u.x": AttackStep(
        asset="c",
        service="u",
        name="x",
        ttc=TTC_HIGH,
        children=["c.u.flag.capture"],
        parents=["c.x"],
    ),
    "c.u.flag.capture": AttackStep(
        asset="c",
        service="u",
        flag="flag",
        name="capture",
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


@pytest.fixture(scope="session")
def graph_yaml(tmpdir_factory):
    graph_yaml = tmpdir_factory.mktemp("test").join("graph.yaml")
    graph_yaml.write(TEST_GRAPH_YAML)
    return graph_yaml


@pytest.fixture(scope="session")
def config_yaml(tmpdir_factory, graph_yaml):
    config_yaml = tmpdir_factory.mktemp("graph_config").join("config.yaml")
    TEST_ENV_CONFIG_YAML["graph_config"]["filename"] = str(graph_yaml)
    to_write = yaml.dump(TEST_ENV_CONFIG_YAML)

    config_yaml.write(to_write)
    return config_yaml


@pytest.fixture(scope="session")
def test_graph_config(config_yaml, graph_yaml):

    config: GraphConfig = GraphConfig.from_yaml(config_yaml)
    config = dataclasses.replace(
        config,
    )

    return config


@pytest.fixture(scope="session")
def test_graph(test_graph_config):
    return AttackGraph(test_graph_config)


@pytest.fixture(scope="session")
def test_env_config(config_yaml):
    config: EnvConfig = EnvConfig.from_yaml(config_yaml)
    dataclasses.replace(config, attacker="well-informed")
    return config


@pytest.fixture(scope="session")
def test_env(test_env_config):
    return AttackSimulationEnv(test_env_config)
