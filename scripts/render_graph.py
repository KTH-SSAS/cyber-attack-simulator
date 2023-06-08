#!/usr/bin/env python3
from attack_simulator.mal.graph import AttackGraph
from attack_simulator.utils.config import GraphConfig

seed = 22
filename = "mgg.yaml"  # sys.argv[1]

config = GraphConfig(filename, rewards={"default": 10}, ttc={"default": 1})

graph = AttackGraph(config)

graph.draw(add_defense=True, width=300, height=400)
