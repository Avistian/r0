"""Exact category-subset model trees and bounded independent split sampling."""

from dataclasses import dataclass, replace

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.category_subset import category_subset_activation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.model_leaf_tree import model_leaf_tree_recipe
from r0.modules.priors.lib.oblique_tree import TreeRoot
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.recursive_forest_records import RecursiveTree
from r0.modules.priors.lib.structural_network import (
    FrozenNetworkFunction,
    NetworkAffine,
    NetworkBlock,
)

_DEFAULT = CausalLimits()


@dataclass(frozen=True, slots=True)
class CategorySubsetSplit:
    feature: int
    categories: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.feature) is not int or self.feature < 0:
            raise ValueError("category split requires a nonnegative integer feature")
        category_subset_activation(self.categories)


def sample_category_subset_splits(
    vocabularies: tuple[tuple[int, ...], ...],
    *,
    count: int,
    inclusion_probability: float = 0.5,
    rng: np.random.Generator,
) -> tuple[CategorySubsetSplit, ...]:
    """Uniform feature slots, then independent Bernoulli inclusion of its codes.

    Empty/full subsets and unused or unseen category codes are retained. This is
    an r0 extension, not a fitted split search or a paper-specific reconstruction.
    """
    if (
        type(vocabularies) is not tuple
        or not 1 <= len(vocabularies) <= 1024
        or type(count) is not int
        or not 1 <= count <= 1024
        or not finite_number(inclusion_probability)
        or not 0 <= inclusion_probability <= 1
    ):
        raise ValueError("invalid bounded category subset sampling controls")
    for vocabulary in vocabularies:
        category_subset_activation(vocabulary)
    with raw_transactions({"splits": rng}, None) as (streams, _):
        active = streams["splits"]
        result: list[CategorySubsetSplit] = []
        for _ in range(count):
            feature = int(active.integers(len(vocabularies)))
            vocabulary = vocabularies[feature]
            included = active.random(len(vocabulary)) < inclusion_probability
            result.append(
                CategorySubsetSplit(
                    feature,
                    tuple(code for code, keep in zip(vocabulary, included, strict=True) if keep),
                )
            )
        return tuple(result)


def categorical_model_tree_recipe(
    task: LinearRecipe,
    splits: tuple[CategorySubsetSplit, ...],
    tree: RecursiveTree,
    models: tuple[NetworkAffine, ...],
    *,
    root: TreeRoot,
    limits: CausalLimits = _DEFAULT,
) -> FunctionalCausalRecipe:
    """Tree feature indices select membership bits; models use original inputs.

    Membership is exactly zero/one. Author thresholds/intervals in that bit space:
    a 0.5 split routes nonmembers left and members right. No ordinal comparison of
    raw category codes is used. Root distributions and leaf models remain explicit.
    """
    if (
        type(splits) is not tuple
        or not 1 <= len(splits) <= 1024
        or any(type(s) is not CategorySubsetSplit or s.feature >= task.features for s in splits)
    ):
        raise ValueError("categorical model tree requires bounded feature subset predicates")
    width = task.features
    count = len(splits)
    parameters = count * width + count * count + sum(len(s.categories) + 2 for s in splits)
    if (
        count > limits.node_channels
        or parameters > limits.parameters
        or 65536 + 512 * parameters > limits.working_bytes
    ):
        raise ValueError("categorical split compilation exceeds limits")
    projection = tuple(tuple(float(j == s.feature) for j in range(width)) for s in splits)
    base = model_leaf_tree_recipe(task, projection, tree, models, root=root, limits=limits)
    blocks = tuple(
        NetworkBlock(
            (0,),
            (
                NetworkAffine(
                    (row,), (0.0,), activation=category_subset_activation(split.categories)
                ),
            ),
        )
        for row, split in zip(projection, splits, strict=True)
    )
    count = len(splits)
    identity = tuple(tuple(float(i == j) for j in range(count)) for i in range(count))
    blocks += (
        NetworkBlock(tuple(range(1, count + 1)), (NetworkAffine(identity, (0.0,) * count),)),
    )
    function = FrozenNetworkFunction(width, blocks)
    nodes = list(base.nodes)
    nodes[1] = replace(nodes[1], function=function)
    return replace(base, nodes=tuple(nodes))
