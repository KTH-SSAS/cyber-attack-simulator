import dataclasses

import pytest

from attack_simulator.config import GraphConfig
from attack_simulator.graph import SIZES, AttackGraph, save_all_default_graphviz


def _assert_same_steps(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == b[key], key


def test_graph_as_expected(test_graph: AttackGraph, test_services, test_attack_steps):
    assert test_graph.service_names == sorted(test_services)
    assert test_graph.attack_names == sorted(test_attack_steps)
    _assert_same_steps(test_graph.attack_steps, test_attack_steps)


def test_graph_same_again(test_graph_config, test_graph):
    g = AttackGraph(test_graph_config)
    assert g.service_names == test_graph.service_names
    assert g.attack_names == test_graph.attack_names
    _assert_same_steps(g.attack_steps, test_graph.attack_steps)


def test_graph_graphviz(test_graph: AttackGraph, test_graph_dot, tmpdir):
    tmp = tmpdir.mkdir("graphviz").join("graph.dot")
    test_graph.save_graphviz(str(tmp), indexed=True)
    assert tmp.read().strip() == test_graph_dot
    # same without an explicit filename
    with tmpdir.as_cwd():
        test_graph.save_graphviz(indexed=True)
        assert tmp.read().strip() == test_graph_dot


def test_graph_prune(test_graph_config: GraphConfig, test_services, test_attack_steps):
    config = dataclasses.replace(test_graph_config, prune=set(["b.v.y"]))
    g = AttackGraph(config)
    assert g.service_names == sorted(set(test_services) - set(["b.v"]))
    expected = set(test_attack_steps.keys()) - set(["b.v.y", "b.v.flag.capture"])
    assert g.attack_names == sorted(expected)
    assert set(g.attack_steps.keys()) == expected


def test_graph_odd_root(test_graph_config: GraphConfig, test_attack_steps):
    config = dataclasses.replace(test_graph_config, root="c.x")
    g = AttackGraph(config)
    assert g.service_names == ["c", "c.u"]
    expected = {step: test_attack_steps[step] for step in ["c.x", "c.u.x", "c.u.flag.capture"]}
    expected["c.x"] = dataclasses.replace(expected["c.x"], parents=[])
    _assert_same_steps(g.attack_steps, expected)


@pytest.mark.parametrize(
    "config",
    [
        GraphConfig(
            low_flag_reward=1,
            medium_flag_reward=2,
            high_flag_reward=3,
            easy_ttc=10,
            hard_ttc=10,
            graph_size="full",
            seed=0,
            filename="graphs/en2720.yaml",
            root="internet.connect",
            unmalleable_assets={"internet", "office_network", "hidden_network"},
        )
    ],
)
def test_graph_config(config):
    g = AttackGraph(config)
    # load full built-in project graph
    assert g.num_services == 18 and g.num_attacks == 78


def test_graph_all_graphviz(test_graph_config, tmpdir):
    with tmpdir.as_cwd():
        save_all_default_graphviz(test_graph_config)
        assert set([i.basename for i in tmpdir.listdir()]) == set([s + ".dot" for s in SIZES])


@pytest.mark.xfail(raises=KeyError)
def test_graph_missing_root(test_graph_config: GraphConfig):
    config = dataclasses.replace(test_graph_config, root="does.not.exist")
    AttackGraph(config)


@pytest.mark.xfail(raises=ValueError)
def test_graph_no_service(test_graph_config, tmpdir):
    yaml = tmpdir.join("bad.yaml")
    yaml.write("---\n\nx: {}\n")
    config = dataclasses.replace(test_graph_config, filename=str(yaml), root="x")
    AttackGraph(config)


# TDOD: verify that different *_TTC and *_FLAG_REWARD values get passed as expected
