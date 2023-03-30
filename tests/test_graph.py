from typing import List

from attack_simulator.mal.graph import AttackGraph


def test_graph_init(
    attack_graph: AttackGraph, test_attack_steps: List[str], test_defense_steps: List[str]
) -> None:
    assert len(attack_graph.attack_steps) == len(test_attack_steps)
    assert len(attack_graph.defense_steps) == len(test_defense_steps)
    # assert len(attack_graph.service_names) == 4
    # assert attack_graph.service_names == sorted(test_services)
    assert attack_graph.attack_names == sorted(test_attack_steps)
