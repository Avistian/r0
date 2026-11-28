"""Replayable bounded r0 portfolio of hard/soft oblique trees with model leaves."""

import math
from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.activation_sampling import finite_number
from r0.modules.priors.lib.categorical_model_tree import (
    CategorySubsetSplit,
    categorical_model_tree_recipe,
    sample_category_subset_splits,
)
from r0.modules.priors.lib.causal_activations import CausalActivation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.mixed_model_tree import (
    NumericProjectionSplit,
    mixed_model_tree_recipe,
    sample_mixed_split_records,
)
from r0.modules.priors.lib.model_leaf_tree import model_leaf_tree_recipe
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.oblique_tree import TreeRoot
from r0.modules.priors.lib.prototype_root_base import PrototypeRootBase
from r0.modules.priors.lib.raw_draw_capture import raw_transactions
from r0.modules.priors.lib.recursive_forest_records import RecursiveTree, RecursiveTreeNode
from r0.modules.priors.lib.sampled_source_root_base import SampledSourceRootBase
from r0.modules.priors.lib.soft_tree import SoftTreeBranch, compile_soft_tree
from r0.modules.priors.lib.source_root_base import SourceRootBase
from r0.modules.priors.lib.structural_network import NetworkAffine
from r0.modules.priors.lib.tree_extension_topology import extension_topology

_DEFAULT = CausalLimits()


@dataclass(frozen=True, slots=True)
class TreeExtensionRequest:
    task: LinearRecipe
    families: tuple[str, ...] = ("hard", "soft")
    leaf_families: tuple[str, ...] = ("constant", "linear", "tanh", "sin")
    min_depth: int = 1
    max_depth: int = 3
    split_density: float = 0.5
    coefficient_scale: float = 1.0
    threshold_bound: float = 3.0
    temperature_bounds: tuple[float, float] = (0.05, 2.0)
    equality: str = "left"
    root: TreeRoot | None = None
    version: str = "r0-tree-extension-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r0-tree-extension-request-v1"
            or type(self.task) is not LinearRecipe
            or type(self.families) is not tuple
            or not self.families
            or any(
                v
                not in (
                    "hard",
                    "soft",
                    "hard-pruned",
                    "soft-pruned",
                    "categorical",
                    "categorical-pruned",
                    "mixed",
                    "mixed-pruned",
                )
                for v in self.families
            )
            or type(self.leaf_families) is not tuple
            or not self.leaf_families
            or any(v not in ("constant", "linear", "tanh", "sin") for v in self.leaf_families)
            or len(self.families) > 64
            or len(self.leaf_families) > 64
            or type(self.min_depth) is not int
            or type(self.max_depth) is not int
            or not 0 <= self.min_depth <= self.max_depth <= 4
            or not finite_number(self.split_density)
            or not 0 <= self.split_density <= 1
            or not finite_number(self.coefficient_scale)
            or not 0 <= self.coefficient_scale <= 1e6
            or not finite_number(self.threshold_bound)
            or not 0 < self.threshold_bound <= 1e6
            or type(self.temperature_bounds) is not tuple
            or len(self.temperature_bounds) != 2
            or any(not finite_number(v) or v <= 0 for v in self.temperature_bounds)
            or self.temperature_bounds[0] > self.temperature_bounds[1]
            or self.equality not in ("left", "right")
        ):
            raise ValueError("invalid bounded tree extension request")

        if self.root is not None and (
            type(self.root)
            not in (JointRootBase, SourceRootBase, SampledSourceRootBase, PrototypeRootBase)
            or self.root.width != self.task.features
        ):
            raise ValueError("tree extension root must match task feature width")

        if any(f.startswith(("categorical", "mixed")) for f in self.families) and (
            type(self.root) is not SourceRootBase
            or self.root.calibration is not None
            or not any(c.family == "categorical" for c in self.root.law.coordinates)
        ):
            raise ValueError(
                "categorical trees require an uncalibrated source root with categorical columns"
            )

        if any(f.startswith("mixed") for f in self.families):
            assert isinstance(self.root, SourceRootBase)
            if not any(c.family != "categorical" for c in self.root.law.coordinates):
                raise ValueError("mixed trees require numeric and categorical root columns")

    def admit(self, limits: CausalLimits) -> None:
        if type(limits) is not CausalLimits:
            raise ValueError("tree extension requires frozen limits")
        metadata, storage = function_record_cost(self, limits)
        f = self.task.features
        o = self.task.classes if self.task.task_type == "multiclass" else 1
        leaves = 2**self.max_depth
        parameters = f * f + leaves * (4 * (f + 2 * o) * (o + 2) + 1024)
        if any(f.startswith(("categorical", "mixed")) for f in self.families):
            assert isinstance(self.root, SourceRootBase)
            cardinality = max(len(c.probabilities) for c in self.root.law.coordinates)
            parameters += leaves * (cardinality + leaves + f)
        if (
            type(limits) is not CausalLimits
            or 4 + 3 * leaves > limits.nodes
            or f + 2 * o > limits.node_channels
            or f + leaves * (1 + 3 * o) > limits.total_channels
            or parameters + metadata > limits.parameters
            or 2**21 + 512 * parameters + 4 * storage > limits.working_bytes
        ):
            raise ValueError("tree extension worst-case construction exceeds limits")


@dataclass(frozen=True, slots=True)
class SampledTreeExtension:
    request: TreeExtensionRequest
    recipe: FunctionalCausalRecipe | NetworkRecipe
    family: str
    depth: int
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    limits: CausalLimits
    version: str = "r0-sampled-tree-extension-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r0-sampled-tree-extension-v1"
            or type(self.request) is not TreeExtensionRequest
            or type(self.recipe) not in (FunctionalCausalRecipe, NetworkRecipe)
            or type(self.depth) is not int
            or not self.request.min_depth <= self.depth <= self.request.max_depth
            or type(self.family) is not str
            or self.family not in self.request.families
            or type(self.initial) is not GeneratorSnapshot
            or type(self.final) is not GeneratorSnapshot
            or type(self.limits) is not CausalLimits
        ):
            raise ValueError("invalid sampled tree extension replay metadata")
        self.request.admit(self.limits)
        self.initial.validate_metadata()
        self.final.validate_metadata()

    def validate_replay(self) -> None:
        self.request.admit(self.limits)
        expected = sample_tree_extension(
            self.request, rng=self.initial.restore(), limits=self.limits
        )
        if self != expected:
            raise ValueError("sampled tree extension differs from retained construction replay")


def sample_tree_extension(
    request: TreeExtensionRequest,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _DEFAULT,
) -> SampledTreeExtension:
    if type(request) is not TreeExtensionRequest:
        raise ValueError("tree extension requires frozen request")
    request.admit(limits)
    with raw_transactions({"construction": rng}, None) as (streams, _):
        active = streams["construction"]
        initial = snapshot_generator(active)
        family = request.families[int(active.integers(len(request.families)))]
        depth = int(active.integers(request.min_depth, request.max_depth + 1))
        children, levels, sides = extension_topology(
            depth, pruned=family.endswith("-pruned"), rng=active
        )
        branches = len(children)
        leaves = len(levels) - branches
        width = request.task.features
        outputs = request.task.classes if request.task.task_type == "multiclass" else 1
        categorical = family.startswith("categorical")
        mixed = family.startswith("mixed")
        mixed_splits: tuple[NumericProjectionSplit | CategorySubsetSplit, ...] = ()
        splits: tuple[CategorySubsetSplit, ...] = ()
        if mixed:
            assert isinstance(request.root, SourceRootBase)
            mixed_splits, mixed_thresholds = sample_mixed_split_records(
                request.root,
                branches,
                request.split_density,
                request.coefficient_scale,
                request.threshold_bound,
                active,
            )
            weights = np.zeros((branches, width))
            thresholds = np.asarray(mixed_thresholds)
            temperatures = np.ones(branches)
        elif categorical:
            assert isinstance(request.root, SourceRootBase)
            eligible = tuple(
                i for i, c in enumerate(request.root.law.coordinates) if c.family == "categorical"
            )
            vocabularies = tuple(
                tuple(range(len(request.root.law.coordinates[i].probabilities))) for i in eligible
            )
            if branches:
                sampled = sample_category_subset_splits(
                    vocabularies,
                    count=branches,
                    inclusion_probability=request.split_density,
                    rng=active,
                )
                splits = tuple(
                    CategorySubsetSplit(eligible[s.feature], s.categories) for s in sampled
                )
            weights = np.zeros((branches, width))
            thresholds = np.full(branches, 0.5)
            temperatures = np.ones(branches)
        else:
            weights = active.normal(0, request.coefficient_scale, (branches, width))
            weights *= active.random((branches, width)) < request.split_density
            thresholds = active.uniform(-request.threshold_bound, request.threshold_bound, branches)
            temperatures = np.exp(
                active.uniform(
                    math.log(request.temperature_bounds[0]),
                    math.log(request.temperature_bounds[1]),
                    branches,
                )
            )
        models: list[NetworkAffine] = []
        for _ in range(leaves):
            kind = request.leaf_families[int(active.integers(len(request.leaf_families)))]
            matrix = active.normal(0, request.coefficient_scale, (outputs, width))
            bias = active.normal(0, request.coefficient_scale, outputs)
            if kind == "constant":
                matrix[:] = 0
            models.append(
                NetworkAffine(
                    tuple(tuple(float(v) for v in row) for row in matrix),
                    tuple(float(v) for v in bias),
                    activation=CausalActivation(kind if kind in ("tanh", "sin") else "identity"),
                )
            )
        projection = tuple(tuple(float(v) for v in row) for row in weights)
        if family in ("soft", "soft-pruned"):
            soft = tuple(
                SoftTreeBranch(
                    projection[i],
                    float(thresholds[i]),
                    children[i][0],
                    children[i][1],
                    float(temperatures[i]),
                )
                for i in range(branches)
            ) + tuple(models)
            recipe: FunctionalCausalRecipe | NetworkRecipe = NetworkRecipe(
                request.task, compile_soft_tree(width, soft, limits=limits), request.root, limits
            )
        else:
            nodes: list[RecursiveTreeNode] = []
            lower, upper = (
                (0.0, 1.0)
                if categorical or mixed_splits and isinstance(mixed_splits[0], CategorySubsetSplit)
                else (-request.threshold_bound, request.threshold_bound)
            )
            for i in range(2 * leaves - 1):
                level = levels[i]
                if i < branches:
                    nodes.append(
                        RecursiveTreeNode(
                            i,
                            float(thresholds[i]),
                            (
                                (0.0, 1.0)
                                if isinstance(mixed_splits[i], CategorySubsetSplit)
                                else (-request.threshold_bound, request.threshold_bound)
                            )
                            if mixed
                            else (lower, upper),
                            level,
                            children[i][0],
                            children[i][1],
                        )
                    )
                else:
                    interval = (
                        (lower, float(thresholds[0]))
                        if branches and sides[i] == 1
                        else (float(thresholds[0]), upper)
                        if branches
                        else (lower, upper)
                    )
                    nodes.append(RecursiveTreeNode(0, interval[0], interval, level, value=(0.0,)))
            if mixed and branches:
                assert request.root is not None
                recipe = mixed_model_tree_recipe(
                    request.task,
                    mixed_splits,
                    RecursiveTree(tuple(nodes)),
                    tuple(models),
                    root=request.root,
                    equality=request.equality,
                    limits=limits,
                )
            elif categorical and branches:
                assert request.root is not None
                recipe = categorical_model_tree_recipe(
                    request.task,
                    splits,
                    RecursiveTree(tuple(nodes)),
                    tuple(models),
                    root=request.root,
                    limits=limits,
                )
            else:
                recipe = model_leaf_tree_recipe(
                    request.task,
                    projection or ((0.0,) * width,),
                    RecursiveTree(tuple(nodes)),
                    tuple(models),
                    equality=request.equality,
                    root=request.root,
                    limits=limits,
                )
        return SampledTreeExtension(
            request, recipe, family, depth, initial, snapshot_generator(active), limits
        )
