from attack_simulator.config import GraphConfig
from attack_simulator.graph import AttackGraph

seed = 22
filename = "graphs/four_ways.yaml"
graph_root = "attacker:13:enter:13"

config = GraphConfig(filename, graph_root, rewards={"default": 10}, ttc={"default": 1})
graph = AttackGraph(config)
graph.draw()
