#!/usr/bin/env python3
from attack_simulator.config import GraphConfig
from attack_simulator.graph import AttackGraph

seed = 22
filename = "mgg.yaml"  # sys.argv[1]

config = GraphConfig(filename, rewards={"default": 10}, ttc={"default": 1})

graph = AttackGraph(config)

graph.draw(add_defense=True, width=300, height=400)
