from dataclasses import asdict
import pygraphviz as pgv
from maltoolbox.attackgraph.node import AttackGraphNode
import json
import sys

dot = sys.stdin.read()

G = pgv.AGraph(dot)

nodes = []
for n in G.nodes_iter():

	node_id = n.attr["label"]
	asset, asset_id, name = node_id.split(":")

	children = list(G.successors(n))
	child_ids = [c.attr["label"] for c in children]

	nodes.append(AttackGraphNode(
		id = node_id,
		asset = asset + ":" + asset_id,
		name = name,
		children = child_ids,
		compromised_by=[],
		type = "or" if name != "defend" else "defense",
	))

nodes = [asdict(n) for n in nodes]

print(json.dumps(nodes, indent=4))


