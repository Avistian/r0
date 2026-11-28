"""Numeric directions and exact category predicates within one model-leaf tree."""

from dataclasses import dataclass, replace

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.categorical_model_tree import CategorySubsetSplit
from r0.modules.priors.lib.category_subset import category_subset_activation
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.model_leaf_tree import model_leaf_tree_recipe
from r0.modules.priors.lib.oblique_tree import TreeRoot
from r0.modules.priors.lib.recursive_forest_records import RecursiveTree
from r0.modules.priors.lib.source_root_base import SourceRootBase
from r0.modules.priors.lib.structural_network import (
    FrozenNetworkFunction,
    NetworkAffine,
    NetworkBlock,
)

_DEFAULT = CausalLimits()


@dataclass(frozen=True, slots=True)
class NumericProjectionSplit:
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.coefficients) is not tuple
            or not 1 <= len(self.coefficients) <= 256
            or any(not finite_number(v) for v in self.coefficients)
        ):
            raise ValueError("numeric split requires bounded finite coefficients")


def mixed_model_tree_recipe(
    task: LinearRecipe,
    splits: tuple[NumericProjectionSplit | CategorySubsetSplit, ...],
    tree: RecursiveTree,
    models: tuple[NetworkAffine, ...],
    *,
    root: TreeRoot,
    equality: str = "left",
    limits: CausalLimits = _DEFAULT,
) -> FunctionalCausalRecipe:
    """Tree feature indices select numeric directions or zero/one membership bits.

    Thresholds/intervals are authored in each predicate's output space. Leaf models
    consume original input columns. Equality applies consistently to both kinds.
    """
    if (
        type(task) is not LinearRecipe
        or type(limits) is not CausalLimits
        or type(splits) is not tuple
        or not 1 <= len(splits) <= 256
        or equality not in ("left", "right")
    ):
        raise ValueError("mixed tree requires bounded frozen split records")
    width, count = task.features, len(splits)
    projection: list[tuple[float, ...]] = []
    activations: list[CausalActivation] = []
    for split in splits:
        if type(split) is NumericProjectionSplit and len(split.coefficients) == width:
            projection.append(split.coefficients)
            activations.append(CausalActivation())
        elif type(split) is CategorySubsetSplit and split.feature < width:
            projection.append(tuple(float(i == split.feature) for i in range(width)))
            activations.append(category_subset_activation(split.categories))
        else:
            raise ValueError("mixed split geometry differs from task features")
    parameters = count * width + count * count + sum(a.parameter_count for a in activations)
    if (
        count > limits.node_channels
        or parameters > limits.parameters
        or 65536 + 512 * parameters > limits.working_bytes
    ):
        raise ValueError("mixed split compilation exceeds limits")
    base = model_leaf_tree_recipe(
        task, tuple(projection), tree, models, root=root, equality=equality, limits=limits
    )
    blocks = tuple(
        NetworkBlock((0,), (NetworkAffine((row,), (0.0,), activation=a),))
        for row, a in zip(projection, activations, strict=True)
    )
    # The right-equality compiler negates thresholds and swaps children. Its
    # predicate values must also be negated, including membership bits.
    sign = -1.0 if equality == "right" else 1.0
    identity = tuple(tuple(sign * float(i == j) for j in range(count)) for i in range(count))
    blocks += (
        NetworkBlock(tuple(range(1, count + 1)), (NetworkAffine(identity, (0.0,) * count),)),
    )
    nodes = list(base.nodes)
    nodes[1] = replace(nodes[1], function=FrozenNetworkFunction(width, blocks))
    return replace(base, nodes=tuple(nodes))


def sample_mixed_split_records(
    root: SourceRootBase,
    count: int,
    density: float,
    scale: float,
    bound: float,
    rng: np.random.Generator,
) -> tuple[tuple[NumericProjectionSplit | CategorySubsetSplit, ...], tuple[float, ...]]:
    """Internal draw primitive; caller validates/admit controls and owns transaction."""
    categorical = tuple(i for i, c in enumerate(root.law.coordinates) if c.family == "categorical")
    numeric = tuple(i for i, c in enumerate(root.law.coordinates) if c.family != "categorical")
    splits: list[NumericProjectionSplit | CategorySubsetSplit] = []
    thresholds: list[float] = []
    for _ in range(count):
        if rng.random() < 0.5:
            feature = categorical[int(rng.integers(len(categorical)))]
            size = len(root.law.coordinates[feature].probabilities)
            included = rng.random(size) < density
            splits.append(
                CategorySubsetSplit(feature, tuple(int(i) for i in np.flatnonzero(included)))
            )
            thresholds.append(0.5)
        else:
            weights = rng.normal(0, scale, len(numeric))
            weights *= rng.random(len(numeric)) < density
            row = [0.0] * root.width
            for feature, value in zip(numeric, weights, strict=True):
                row[feature] = float(value)
            splits.append(NumericProjectionSplit(tuple(row)))
            thresholds.append(float(rng.uniform(-bound, bound)))
    return tuple(splits), tuple(thresholds)
