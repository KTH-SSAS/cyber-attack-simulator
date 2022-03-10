from typing import Any, Dict, Tuple


def tree_layout(root: Any, children: Dict[Any, Any]) -> Dict[Any, Tuple[float, float]]:
    """Adapted from http://e-archive.informatik.uni-
    koeln.de/431/1/zaik2002-431.ps."""

    distance = 1
    changes = {node: 0 for node in children}
    shifts = {node: 0 for node in children}
    mod = {node: 0 for node in children}
    thread = {node: None for node in children}
    ancestor = {node: node for node in children}
    prelim: dict = {}
    level = {}
    pos = {}
    child_count = {node: len(children[node]) for node in children}
    parent = {child: node for node in children for child in children[node]}

    def next_left(node):
        return thread[node] if not child_count[node] else children[node][0]

    def next_right(node):
        return thread[node] if not child_count[node] else children[node][-1]

    def apportion(node, default_ancestor, left_sibling):
        if left_sibling is not None:
            siblings = children[parent[node]]

            right_in = right_out = node
            left_in = left_sibling
            left_out = default_ancestor

            mod_right_in = mod[right_in]
            mod_right_out = mod[right_out]
            mod_left_in = mod[left_in]
            mod_left_out = mod[left_out]

            next_left_in = next_right(left_in)
            next_right_in = next_left(right_in)
            next_left_out = next_left(left_out)
            next_right_out = next_right(right_out)

            while (
                next_left_in is not None
                and next_right_in is not None
                and next_left_out is not None
                and next_right_out is not None
            ):
                left_in = next_left_in
                right_in = next_right_in
                left_out = next_left_out
                right_out = next_right_out

                ancestor[right_out] = node
                shift = (
                    (prelim[left_in] + mod_left_in) - (prelim[right_in] + mod_right_in) + distance
                )
                if 0 < shift:
                    greatest_ancestor = (
                        default_ancestor if ancestor[left_in] not in siblings else ancestor[left_in]
                    )

                    subtrees = siblings.index(node) - siblings.index(greatest_ancestor)
                    if subtrees:
                        delta = shift / subtrees

                        changes[node] += -delta
                        shifts[node] += shift
                        prelim[node] += shift
                        mod[node] += shift

                        changes[greatest_ancestor] += delta

                    mod_right_in += shift
                    mod_right_out += shift

                mod_right_in += mod[right_in]
                mod_right_out += mod[right_out]
                mod_left_in += mod[left_in]
                mod_left_out += mod[left_out]

                next_left_in = next_right(left_in)
                next_right_in = next_left(right_in)
                next_left_out = next_left(left_out)
                next_right_out = next_right(right_out)

                if next_left_in is not None and next_right_out is None:
                    thread[right_out] = next_left_in
                    mod[right_out] += mod_left_in - mod_right_out

                if next_right_in is not None and next_left_out is None:
                    thread[left_out] = next_right_in
                    mod[left_out] += mod_right_in - mod_left_out

                    default_ancestor = node

    def first_walk(node, depth=0, left_sibling=None):
        level[node] = depth
        if child_count[node] == 0:
            prelim[node] = 0 if left_sibling is None else prelim[left_sibling] + distance
        else:
            leftmost = children[node][0]
            rightmost = children[node][-1]

            child_left_sibling = None
            for child in children[node]:
                first_walk(child, depth + 1, child_left_sibling)
                apportion(child, leftmost, left_sibling)
                child_left_sibling = child

            shift = change = 0
            for child in reversed(children[node]):
                prelim[child] += shift
                mod[child] += shift
                change += changes[child]
                shift += shifts[child] + change

            midpoint = (prelim[leftmost] + prelim[rightmost]) / 2
            if left_sibling is not None:
                prelim[node] = prelim[left_sibling] + distance
                mod[node] = prelim[node] - midpoint
            else:
                prelim[node] = midpoint

    def second_walk(node, shift):
        pos[node] = (level[node], prelim[node] + shift)
        for child in children[node]:
            second_walk(child, shift + mod[node])

    first_walk(root)
    second_walk(root, -prelim[root])

    return pos
