import logging
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np

from .graph import AttackGraph
from .tree_layout import tree_layout
from .tweak_layout import tweak_children

logger = logging.getLogger("simulator")


def nx_digraph(g: AttackGraph, indices=True) -> nx.DiGraph:
    dig = nx.DiGraph()
    if indices:
        dig.add_nodes_from(range(g.num_attacks))
        dig.add_edges_from(
            [
                (attack_index, child_index)
                for attack_index in range(g.num_attacks)
                for child_index in g.child_indices[attack_index]
            ]
        )
    else:
        dig.add_nodes_from(g.attack_names)
        dig.add_edges_from(
            [(name, child) for name in g.attack_names for child in g.attack_steps[name].children]
        )
    return dig


def _handle_unassigned(
    g: nx.Graph, root: Any, pos: Dict[Any, Tuple[float, float]]
) -> Dict[Any, Tuple[float, float]]:
    unassigned = g.nodes - pos

    if unassigned:
        size = len(unassigned)
        total = len(g.nodes)
        logger.warn(f"Generating random position(s) for {size} node(s) not connected to '{root}'")
        (xmin, xmax), (ymin, ymax) = tuple(map(lambda l: (min(l), max(l)), zip(*pos.values())))
        dx = xmax - xmin
        dy = ymax - ymin
        if dx <= dy:
            xmin = xmax
            xmax += dx * size / total
        else:
            ymin = ymax
            ymax += dy * size / total
        for node in unassigned:
            pos[node] = (np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))

    return pos


def nx_dag_layout(g: nx.DiGraph, tweak=None, debug=False) -> Dict[Any, Tuple[float, float]]:
    # pick a root
    # TODO: handle multiple roots better
    roots = [node for node, in_degree in g.in_degree if in_degree == 0]
    if not roots:
        logger.warn("No node with zero in-degree: picking a random node as root.")
        root = np.random.choice(g.nodes)
    elif 1 < len(roots):
        logger.warn("Multiple nodes with zero in-degree: picking one at random.")
        root = np.random.choice(roots)
    else:
        root = roots[0]

    # determine children by level
    children = {}
    level = {}
    depth = 0
    nodes = set((root,))
    while nodes:
        next_nodes = set()
        for node in nodes:
            level[node] = depth
            children[node] = list(g.successors(node))
            next_nodes |= set(children[node])
        depth += 1
        nodes = next_nodes

    # ignore children on lower levels (DAG --> tree)
    tree_children = {
        node: [child for child in children[node] if level[child] == level[node] + 1]
        for node in children
    }

    pos = tree_layout(root, tree_children)

    if tweak:
        skip_edges = [
            (node, child)
            for node in children
            for child in (set(children[node]) - set(tree_children[node]))
        ]

        if len(skip_edges):
            if isinstance(tweak, (int, float)):
                tweak = (tweak,)
            if not isinstance(tweak, tuple) or len(tweak) == 0:
                tweak = (depth ** 2,)
            if len(tweak) == 1:
                tweak += (tweak[0] ** 2,)

            if tweak_children(pos, root, tree_children, skip_edges, tweak):
                pos = tree_layout(root, tree_children)

    return _handle_unassigned(g, root, pos)
