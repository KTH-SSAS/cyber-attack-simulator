#!python
import networkx as nx
import json
import sys

graph_data = json.loads(sys.stdin.read())

children = {
	n["id"] : [c for c in n["children"]] for n in graph_data
}

edges = [
	(f"\\\"{n}\\\"", f"\\\"{c}\\\"") for n in children for c in children[n]
]

G = nx.DiGraph()
G.add_edges_from(edges)

# export to json

data = nx.readwrite.json_graph.node_link_data(G)

sys.stdout.write(json.dumps(data) + "\n")