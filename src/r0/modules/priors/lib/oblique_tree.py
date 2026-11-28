"""Exact hard oblique forest tasks compiled to projection and existing traversal."""

from dataclasses import replace

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.recursive_forest_function import FrozenRecursiveForestFunction
from r0.modules.priors.lib.recursive_forest_records import RecursiveForest, RecursiveTree
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

_DEFAULT = CausalLimits()


def oblique_tree_recipe(
    task: LinearRecipe,
    projection: Matrix,
    forest: RecursiveForest,
    *,
    equality: str = "left",
    root: JointRootBase | None = None,
    limits: CausalLimits = _DEFAULT,
) -> FunctionalCausalRecipe:
    """Observed raw features -> linear split directions -> hard forest scores.

    Projection rows may be sparse and may share raw features. Forest feature
    indices select these directions. Regression-vector leaves are scores for the
    ordinary task adapter; native classifier labels are intentionally rejected.
    """
    height, width = matrix_shape(projection)
    if (
        type(task) is not LinearRecipe
        or type(forest) is not RecursiveForest
        or type(limits) is not CausalLimits
        or equality not in ("left", "right")
        or forest.task != "regression"
        or forest.input_width != height
        or task.features != width
        or forest.output_width != (task.classes if task.task_type == "multiclass" else 1)
        or root is not None
        and (type(root) is not JointRootBase or root.width != width)
    ):
        raise ValueError("oblique forest geometry, score semantics or equality rule differs")
    forest.admit(task.rows, limits)
    limits.admit_dimensions(
        (width, height, forest.output_width),
        ((), (0,), (1,)),
        rows=task.rows,
        calibration_rows=0,
        features=width,
        outputs=forest.output_width,
        extra_parameters=width * width + height * width,
    )
    if equality == "right":
        # Negating each direction and threshold, then exchanging children, makes
        # the existing <= traversal implement original strict-left comparison.
        projection = tuple(tuple(-v for v in row) for row in projection)
        forest = replace(
            forest,
            trees=tuple(
                RecursiveTree(
                    tuple(
                        replace(
                            n,
                            threshold=-n.threshold,
                            interval=(-n.interval[1], -n.interval[0]),
                            left=n.right,
                            right=n.left,
                        )
                        for n in tree.nodes
                    )
                )
                for tree in forest.trees
            ),
        )
    identity = tuple(tuple(float(i == j) for j in range(width)) for i in range(width))
    return FunctionalCausalRecipe(
        task,
        (
            FunctionalCausalNode(
                (), FrozenLinearFunction(identity), root or JointRootBase("gaussian", width)
            ),
            FunctionalCausalNode((0,), FrozenLinearFunction(projection)),
            FunctionalCausalNode((1,), FrozenRecursiveForestFunction(forest)),
        ),
        tuple((0, j) for j in range(width)),
        tuple((2, j) for j in range(forest.output_width)),
        limits,
    )
