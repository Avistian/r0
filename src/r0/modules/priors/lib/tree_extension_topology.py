"""Bounded tree shapes with internal nodes before terminal models."""

import numpy as np


def extension_topology(
    depth: int, *, pruned: bool, rng: np.random.Generator
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    """Keep a depth-length left spine; independently prune other branches at 1/2.

    Each eligible off-spine node draws once, in breadth-first order. The spine
    guarantees the requested maximum depth. Balanced mode consumes no randomness.
    Return internal children, all node depths, and each node's root-child side.
    """
    pending = [(0, 0, True)]
    children: dict[int, tuple[int, int]] = {}
    cursor = 0
    while cursor < len(pending):
        level, side, spine = pending[cursor]
        split = level < depth and (not pruned or spine or rng.random() < 0.5)
        if split:
            left = len(pending)
            children[cursor] = (left, left + 1)
            pending.extend(
                (
                    (level + 1, side or 1, spine),
                    (level + 1, side or 2, False),
                )
            )
        cursor += 1
    order = list(children) + [i for i in range(len(pending)) if i not in children]
    mapping = {old: new for new, old in enumerate(order)}
    return (
        tuple((mapping[left], mapping[right]) for left, right in children.values()),
        tuple(pending[i][0] for i in order),
        tuple(pending[i][1] for i in order),
    )
