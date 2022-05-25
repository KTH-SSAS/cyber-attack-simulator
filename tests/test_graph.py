import dataclasses
from typing import Dict

import pytest

from attack_simulator.config import GraphConfig
from attack_simulator.graph import AttackGraph, AttackStep


def _assert_same_steps(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == b[key], key


def test_graph_as_expected(attack_graph: AttackGraph, test_services, test_attack_steps):
    assert len(attack_graph.attack_steps) == 6
    assert len(attack_graph.defense_steps) == 2
    assert len(attack_graph.service_names) == 4
    assert attack_graph.service_names == sorted(test_services)
    assert attack_graph.attack_names == sorted(test_attack_steps)
    _assert_same_steps(attack_graph.attack_steps, test_attack_steps)


def test_graph_rewards(attack_graph: AttackGraph, test_attack_steps: Dict[str, AttackStep]):
    
    for step in test_attack_steps:
        reward = attack_graph.reward_params[attack_graph.attack_indices[step]]
        assert reward == test_attack_steps[step].reward

def test_graph_same_again(graph_config, attack_graph):
    g = AttackGraph(graph_config)
    assert g.service_names == attack_graph.service_names
    assert g.attack_names == attack_graph.attack_names
    _assert_same_steps(attack_graph.attack_steps, g.attack_steps)


@pytest.mark.xfail(
    reason="The logic of the 'prune' arg needs to be updated with the new instance model"
)
def test_graph_prune(graph_config: GraphConfig, test_services, test_attack_steps):
    config = dataclasses.replace(graph_config, prune=["b.v"])
    g = AttackGraph(config)
    assert g.service_names == sorted(set(test_services) - {"b.v"})
    expected = set(test_attack_steps.keys()) - {"b.v.y", "b.v.flag.capture"}
    assert g.attack_names == sorted(expected)
    assert set(g.attack_steps.keys()) == expected


@pytest.mark.xfail(reason="The logic of this needs to be updated with the new instance model")
def test_graph_odd_root(graph_config: GraphConfig, test_attack_steps):
    config = dataclasses.replace(graph_config, root="c.x")
    g = AttackGraph(config)
    assert g.service_names == ["c", "c.u"]
    expected = {step: test_attack_steps[step] for step in ["c.x", "c.u.x", "c.u.flag.capture"]}
    expected["c.x"] = dataclasses.replace(expected["c.x"], parents=[])
    _assert_same_steps(g.attack_steps, expected)


@pytest.mark.xfail(raises=KeyError)
def test_graph_missing_root(graph_config: GraphConfig):
    config = dataclasses.replace(graph_config, root="does.not.exist")
    AttackGraph(config)
