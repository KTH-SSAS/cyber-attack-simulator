from collections import defaultdict

import numpy as np


def _tweak_children(
    pos, root, children, skip_edges, intersection_weights, generations=3, attempts_per_generation=13
):
    """Tweak _order_ of children to center the graph and reduce edge lengths and crossings

    Swap predetermined vertex positions leaving the total **in-tree** edge length unchanged.
    Only lengths of edges that _skip_ between levels can change.
    Similarly, only intersections that involve edges that _skip_ between levels are relevant.
    For centering, the deviation from the mid-line at zero is penalized.

    Returns True when a better permutation is found, False otherwise.
    (`children` manipulated in-place as a side effect).
    """
    original_pos = pos.copy()
    counts = {}
    sizes = {}
    costs = {}
    skips = {}
    weights = {}
    excludes = {}

    def edge_length(u, v):
        return np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))

    def intersection_penalty(u0, v0, u1, v1):
        p0 = np.array(pos[u0])
        d0 = np.array(pos[v0]) - p0
        p1 = np.array(pos[u1])
        d1 = np.array(pos[v1]) - p1
        dp = p1 - p0

        cross_ = np.cross(d0, d1)
        cross0 = np.cross(dp, d0)
        cross1 = np.cross(dp, d1)

        if cross_:  # not parallel nor co-linear
            return (
                intersection_weights[0]
                if 0 < cross0 / cross_ < 1 and 0 < cross1 / cross_ < 1
                else 0
            )
        elif cross0 or cross1:  # parallel
            return 0
        else:  # co-linear
            dot00 = np.dot(d0, d0)
            dot01 = np.dot(d0, d1)
            dot0p = np.dot(d0, dp)
            return intersection_weights[1] if 0 < dot0p < dot00 or 0 < dot0p + dot01 < dot00 else 0

    def edge_penalty(node):
        penalty = 0
        for child in children[node]:
            for skip_node, skip_child in skip_edges:
                skip_cost = edge_length(skip_node, skip_child) * intersection_penalty(
                    skip_node, skip_child, node, child
                )
                skips[skip_node] += skip_cost
                penalty += skip_cost
        return penalty

    def move_subtree(node, shift):
        # update position
        x, y = pos[node]
        y += shift
        pos[node] = (x, y)
        for child in children[node]:
            move_subtree(child, shift)

    def count(node):
        counts[node] = len(children[node])
        sizes[node] = size = 1 + sum([count(child) for child in children[node]])
        return size

    def evaluate(node):
        # vertical deviation from mid-line (at zero) + edge penalty + costs for children
        cost = (
            abs(pos[node][1])
            + edge_penalty(node)
            + sum([evaluate(child) for child in children[node]])
        )
        costs[node] = cost
        return cost

    def incorporate_skips(node):
        skip_cost = skips[node] + sum([incorporate_skips(child) for child in children[node]])
        costs[node] += skip_cost
        return skip_cost

    def calculate_cost():
        nonlocal skips

        skips = defaultdict(float)
        cost = evaluate(root)
        cost += incorporate_skips(root)
        assert cost == costs[root]
        for node in costs:
            weights[node] = 0 if counts[node] < 2 else costs[node] / sizes[node]
        return cost

    def try_random_swap(node, best):
        nonlocal pos

        # short-hand
        c = children[node]
        s = set(c)

        # pick random children to swap
        pool_a = s.copy()
        while pool_a:
            a = np.random.choice(list(pool_a))
            pool_b = s - excludes[a]
            if pool_b:
                break
            pool_a -= {a}

        if not pool_a:
            weights[node] = 0
            return best

        b = np.random.choice(list(pool_b))

        excludes[a] |= {b}
        excludes[b] |= {a}

        # save current configuration
        pos_ = pos.copy()

        diff = pos[b][1] - pos[a][1]
        move_subtree(a, diff)
        move_subtree(b, -diff)
        cost = calculate_cost()

        if cost < best:
            best = cost
            # swap child order
            i = c.index(a)
            j = c.index(b)
            c[i] = b
            c[j] = a
        else:
            # revert positions
            pos = pos_

        return best

    def normalize(values):
        return np.array(values) / np.sum(values)

    count(root)
    previous_best = calculate_cost()
    while generations and np.any(weights.values()):
        excludes = {node: {node} for node in children}
        success = 0
        for i in range(attempts_per_generation):
            node = np.random.choice(list(weights.keys()), p=normalize(list(weights.values())))
            best = try_random_swap(node, previous_best)
            if best < previous_best:
                success += 1
                previous_best = best
        if not success:
            generations -= 1

    return pos != original_pos
