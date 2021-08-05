import pytest

from attack_simulator.graph import AttackGraph, AttackStep

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
  reward: EARLY_FLAG_REWARD
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
  reward: LATE_FLAG_REWARD
c.x:
  children:
    - c.u.x
  step_type: and
c.u.x:
  ttc: HARD_TTC
  children:
    - c.u.flag.capture
c.u.flag.capture:
  reward: FINAL_FLAG_REWARD
"""

TEST_SERVICES = ["a", "b", "b.u", "b.v", "c", "c.u"]

TEST_ATTACK_STEPS = {
    "a.x": AttackStep(asset="a", name="x", children=("b.x",)),
    "b.x": AttackStep(
        asset="b",
        name="x",
        ttc=10,
        children=(
            "b.flag.capture",
            "b.u.y",
            "b.v.y",
        ),
        parents=("a.x",),
    ),
    "b.flag.capture": AttackStep(
        asset="b",
        flag="flag",
        name="capture",
        reward=10000,
        parents=("b.x",),
    ),
    "b.u.y": AttackStep(
        asset="b",
        service="u",
        name="y",
        ttc=10,
        children=("c.x",),
        parents=("b.x",),
    ),
    "b.v.y": AttackStep(
        asset="b",
        service="v",
        name="y",
        ttc=100,
        children=("b.v.flag.capture", "c.x"),
        parents=("b.x",),
    ),
    "b.v.flag.capture": AttackStep(
        asset="b",
        service="v",
        flag="flag",
        name="capture",
        reward=10000,
        parents=("b.v.y",),
    ),
    "c.x": AttackStep(
        asset="c",
        name="x",
        step_type="and",
        children=("c.u.x",),
        parents=("b.u.y", "b.v.y"),
    ),
    "c.u.x": AttackStep(
        asset="c",
        service="u",
        name="x",
        ttc=100,
        children=("c.u.flag.capture",),
        parents=("c.x",),
    ),
    "c.u.flag.capture": AttackStep(
        asset="c",
        service="u",
        flag="flag",
        name="capture",
        reward=10000,
        parents=("c.u.x",),
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
def test_graph_config(tmpdir_factory):
    graph_yaml = tmpdir_factory.mktemp("test").join("graph.yaml")
    graph_yaml.write(TEST_GRAPH_YAML)
    return dict(filename=str(graph_yaml), root="a.x", random_seed=42)


@pytest.fixture(scope="session")
def test_graph(test_graph_config):
    return AttackGraph(test_graph_config)
