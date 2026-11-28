"""Hard-routed model leaves compiled into existing frozen graph operations."""

from dataclasses import replace

from r0.modules.priors.lib.category_subset import category_subset_activation
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_primitives import FrozenLinearFunction
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline
from r0.modules.priors.lib.oblique_tree import TreeRoot, oblique_tree_recipe
from r0.modules.priors.lib.recursive_forest_records import RecursiveForest, RecursiveTree
from r0.modules.priors.lib.structural_network import (
    FrozenNetworkFunction,
    NetworkAffine,
    NetworkBlock,
)
from r0.modules.priors.lib.weight_geometry import Matrix, matrix_shape

_DEFAULT = CausalLimits()


def model_leaf_tree_recipe(
    task: LinearRecipe,
    projection: Matrix,
    tree: RecursiveTree,
    models: tuple[NetworkAffine, ...],
    *,
    equality: str = "left",
    root: TreeRoot | None = None,
    limits: CausalLimits = _DEFAULT,
) -> FunctionalCausalRecipe:
    """Models follow terminal-node order, independent of left/right traversal order.

    Original terminal values/labels are replaced by model indices. All models
    evaluate raw input features. Nonfinite inactive models still fail explicitly.
    """
    directions, inputs = matrix_shape(projection)
    if (
        type(task) is not LinearRecipe
        or type(tree) is not RecursiveTree
        or type(limits) is not CausalLimits
    ):
        raise ValueError("model tree requires frozen task, tree and limits")
    terminals = tuple(i for i, n in enumerate(tree.nodes) if n.left == -1)
    outputs = task.classes if task.task_type == "multiclass" else 1
    if (
        type(models) is not tuple
        or len(models) != len(terminals)
        or any(
            type(m) is not NetworkAffine or (m.input_width, m.output_width) != (inputs, outputs)
            for m in models
        )
        or task.features != inputs
    ):
        raise ValueError("model leaves must match terminal count and task geometry")
    count = len(models)
    parameters = (
        sum(m.parameter_count for m in models)
        + count * (outputs * outputs + outputs + 8)
        + outputs * outputs * count
    )
    if (
        4 + 3 * count > limits.nodes
        or inputs + directions + 1 + count * (1 + 2 * outputs) + outputs > limits.total_channels
        or parameters > limits.parameters
        or 65536 + 512 * parameters > limits.working_bytes
    ):
        raise ValueError("model tree compiled construction exceeds limits")
    ids = {node: i for i, node in enumerate(terminals)}
    selector = RecursiveTree(
        tuple(
            replace(n, value=(float(ids[i]),), label=None) if i in ids else n
            for i, n in enumerate(tree.nodes)
        )
    )
    base_task = replace(task, task_type="regression", classes=0, class_offsets=())
    base = oblique_tree_recipe(
        base_task,
        projection,
        RecursiveForest(directions, (selector,)),
        equality=equality,
        root=root,
        limits=limits,
    )
    nodes = list(base.nodes)
    contributions: list[int] = []
    identity = tuple(tuple(float(i == j) for j in range(outputs)) for i in range(outputs))
    for i, model in enumerate(models):
        model_index = len(nodes)
        nodes.append(
            FunctionalCausalNode(
                (0,), FrozenNetworkFunction(inputs, (NetworkBlock((0,), (model,)),))
            )
        )
        mask_index = len(nodes)
        nodes.append(
            FunctionalCausalNode(
                (2,),
                FrozenLinearFunction(((1.0,),)),
                pipeline=CausalNodePipeline(1, (category_subset_activation((i,)),)),
            )
        )
        contributions.append(len(nodes))
        nodes.append(
            FunctionalCausalNode(
                (mask_index, model_index),
                FrozenParentFunction(
                    (FrozenLinearFunction(((1.0,),) * outputs), FrozenLinearFunction(identity)),
                    aggregation="product",
                ),
            )
        )
    output_index = len(nodes)
    reduction = tuple(
        tuple(float(j % outputs == i) for j in range(outputs * count)) for i in range(outputs)
    )
    nodes.append(FunctionalCausalNode(tuple(contributions), FrozenLinearFunction(reduction)))
    return FunctionalCausalRecipe(
        task, tuple(nodes), base.observed, tuple((output_index, j) for j in range(outputs)), limits
    )
