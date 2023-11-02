import json
import networkx as nx
import matplotlib.pyplot as plt


log_file = "obs_log.jsonl"

with open(log_file, "r") as f:
    lines = f.readlines()

lines = [json.loads(line) for line in lines]

ax1: plt.Axes
ax2: plt.Axes
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.ion()
plt.show()
for line in lines:

    graph = nx.DiGraph()
    edges = line["obs"]["defender"]["edges"]
    edges = [(e1, e2) for e1, e2 in zip(edges[0], edges[1])]
    graph.add_edges_from(edges)

    defense_nodes = line["obs"]["defender"]["defense_indices"]
    action_dist = line["action_distributions"]["defender"]
    node_states = line["obs"]["defender"]["observation"]
    colors = ["r" if i == 1 and (i not in defense_nodes) else "b" for i in node_states]

    layout = nx.nx_pydot.graphviz_layout(graph, prog="dot")
    nx.draw_networkx(graph, pos=layout, with_labels=True, arrows=True, ax=ax1, node_color=colors)
    ax2.hist(action_dist)

    plt.draw()
    plt.pause(1)
    ax1.clear()
    ax2.clear()

    print(line["actions"])
