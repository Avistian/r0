"""Exact class-label emission through a deterministic indicator-to-category boundary."""

from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_root_base import JointRootBase
from r0.modules.priors.lib.native_causal import NativeCausalNode, VariableDomain
from r0.modules.priors.lib.prototype_root_base import PrototypeRootBase
from r0.modules.priors.lib.recursive_forest_function import FrozenRecursiveForestFunction
from r0.modules.priors.lib.source_root_base import SourceRootBase


def recursive_forest_native_node(
    function: FrozenRecursiveForestFunction,
    *,
    parents: tuple[int, ...] = (),
    root: JointRootBase | SourceRootBase | PrototypeRootBase | None = None,
    vocabulary: tuple[str, ...] = (),
) -> NativeCausalNode:
    """No noise/pipeline or stochastic softmax: retain the forest's actual majority label."""
    if type(function) is not FrozenRecursiveForestFunction:
        raise ValueError("native recursive node requires its frozen structural envelope")
    if type(vocabulary) is not tuple:
        raise ValueError("recursive vocabulary must be an immutable tuple")
    if function.forest.task == "classification":
        if len(vocabulary) != function.forest.classes:
            raise ValueError("recursive class vocabulary must cover each declared label")
        domain = VariableDomain(
            "binary" if len(vocabulary) == 2 else "categorical", vocabulary, "argmax"
        )
    else:
        if vocabulary:
            raise ValueError("regression recursive nodes have no class vocabulary")
        domain = VariableDomain()
    equation = FunctionalCausalNode(
        parents,
        function,
        root,
        version="functional-causal-node-v2"
        if isinstance(root, (SourceRootBase, PrototypeRootBase))
        else "functional-causal-node-v1",
    )
    return NativeCausalNode(equation, domain)
