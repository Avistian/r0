"""Strict complete recursive records, admitted before any construction RNG replay."""

from dataclasses import fields
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_tree_decoding import fitted_array, fitted_object
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.recursive_forest_plan import RecursiveForestPlan
from r0.modules.priors.lib.recursive_forest_records import (
    RecursiveForest,
    RecursiveTree,
    RecursiveTreeNode,
)
from r0.modules.priors.lib.recursive_forest_sampling import (
    SampledRecursiveForest,
    sample_recursive_forest,
)
from r0.modules.priors.lib.sampled_graph_decoding import construction_matches

_DEFAULT_LIMITS = CausalLimits()


def _complete(data: Any, kind: type[Any]) -> dict[str, Any]:
    record = fitted_object(data, kind)
    if set(record) != {f.name for f in fields(kind)}:
        raise ValueError("recursive archive requires complete exact fields")
    return record


def recursive_forest_plan_from_dict(data: Any) -> RecursiveForestPlan:
    record = _complete(data, RecursiveForestPlan)
    record["intervals"] = tuple(fitted_array(p, 2) for p in fitted_array(record["intervals"], 256))
    for key in ("leaf_mean", "leaf_scale", "split_probabilities"):
        record[key] = fitted_array(record[key], 256)
    return RecursiveForestPlan(**record)


def recursive_forest_from_dict(
    data: Any,
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> RecursiveForest:
    if type(limits) is not CausalLimits:
        raise ValueError("recursive decoder requires frozen limits")
    admit_function_metadata(data, limits)
    record = _complete(data, RecursiveForest)
    width, output = record["input_width"], record["output_width"]
    if (
        type(width) is not int
        or type(output) is not int
        or not 1 <= width <= 256
        or not 1 <= output <= 32
        or max(width, output) > limits.node_channels
        or width + output > limits.total_channels
    ):
        raise ValueError("recursive archive channel limit exceeded")
    raw_trees = fitted_array(record["trees"], 64)
    raw_nodes: list[Any] = []
    count = 0
    for tree in raw_trees:
        nodes = _complete(tree, RecursiveTree)["nodes"]
        if type(nodes) not in (list, tuple) or not 1 <= len(nodes) <= 2047:
            raise ValueError("recursive archive node array differs")
        count += len(nodes)
        if count > limits.nodes:
            raise ValueError("recursive archive node limit exceeded")
        raw_nodes.append(nodes)
    trees: list[RecursiveTree] = []
    for raw in raw_nodes:
        nodes: list[RecursiveTreeNode] = []
        for value in raw:
            node = _complete(value, RecursiveTreeNode)
            node["interval"] = fitted_array(node["interval"], 2)
            node["value"] = fitted_array(node["value"], 32)
            nodes.append(RecursiveTreeNode(**node))
        trees.append(RecursiveTree(tuple(nodes)))
    record["trees"] = tuple(trees)
    result = RecursiveForest(**record)
    result.admit(1, limits)
    return result


def sampled_recursive_forest_from_dict(
    data: Any,
    *,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> SampledRecursiveForest:
    if type(limits) is not CausalLimits:
        raise ValueError("recursive decoder requires frozen limits")
    admit_function_metadata(data, limits)
    record = _complete(data, SampledRecursiveForest)
    declared = CausalLimits(**_complete(record["limits"], CausalLimits))
    effective = CausalLimits(
        **{
            f.name: min(getattr(limits, f.name), getattr(declared, f.name))
            for f in fields(CausalLimits)
        }
    )
    raw_limits = RawDrawLimits(**_complete(record["raw_limits"], RawDrawLimits))
    plan = recursive_forest_plan_from_dict(record["plan"])
    plan.admit(effective, raw_limits)
    forests = fitted_array(record["forests"], plan.populations)
    if len(forests) != plan.populations:
        raise ValueError("recursive population archive shape differs")
    # Geometry is cheap to inspect and must not be deferred until after reconstruction.
    for forest in forests:
        item = recursive_forest_from_dict(forest, limits=effective)
        if (
            item.input_width != plan.input_width
            or item.output_width != plan.output_width
            or len(item.trees) != plan.estimators
            or item.task != plan.task
        ):
            raise ValueError("recursive forest archive geometry differs from plan")
    fitted_array(record["raw"], 4 * plan.worst_nodes + plan.populations * plan.estimators)
    initial = GeneratorSnapshot(**_complete(record["initial"], GeneratorSnapshot))
    rng = initial.restore(byte_limit=raw_limits.state_bytes)
    result = sample_recursive_forest(plan, rng=rng, limits=declared, raw_limits=raw_limits)
    if not construction_matches(result, data):
        raise ValueError("recursive archive differs from complete exact replay")
    return result
