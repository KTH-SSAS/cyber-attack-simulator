#!/usr/bin/env python3
from attack_simulator.mal.graph import AttackGraph
from attack_simulator.utils.config import GraphConfig

seed = 22
filename = "test_atkgraph_10_51_21.yaml"  # sys.argv[1]

config = GraphConfig(filename, rewards={"default": 10}, ttc={"default": 1})

graph = AttackGraph(config)

height = 600
width = height * 1.618

graph.draw(add_defense=True, width=width, height=height)

print(len(graph.attack_names))
print(len(graph.defense_names))
