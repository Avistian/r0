"""Array-free whole-task geometry before any sampled forest receipt is replayed."""

from dataclasses import fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_decoding import decode_admitted_function
from r0.modules.priors.lib.frozen_function_evaluation import function_geometry
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.functional_causal_decoding import functional_pipeline
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.recursive_forest_function import (
    FrozenRecursiveForestFunction,
    recursive_function_members,
)
from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits
from r0.modules.priors.lib.recursive_root_preflight import (
    recursive_root_budget,
    recursive_root_width,
)

RECURSIVE_TASK_VERSIONS = frozenset(
    (
        "functional-causal-v1",
        "functional-causal-v2",
        "fitted-functional-causal-v1",
        "sampled-fitted-functional-causal-v1",
        "population-fitted-functional-causal-v1",
        "sampled-population-fitted-functional-causal-v1",
        "support-population-functional-causal-v1",
        "native-causal-v1",
        "native-vector-causal-v1",
        "converted-causal-v1",
    )
)


def _object(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("recursive task admission requires graph objects")
    return cast(dict[str, Any], value)


def _array(value: Any) -> list[Any] | tuple[Any, ...]:
    if type(value) not in (tuple, list):
        raise ValueError("recursive task admission requires graph arrays")
    return cast(list[Any] | tuple[Any, ...], value)


def _authored(value: Any) -> Any:
    """Only geometry is decoded here. Original receipts remain untouched for strict replay."""
    if type(value) is dict:
        raw = _object(value)
        if raw.get("version") == "frozen-recursive-forest-function-v1":
            source = _object(raw.get("source"))
            if source.get("version") == "sampled-recursive-forest-v1":
                forests = _array(source.get("forests"))
                if len(forests) != 1:
                    raise ValueError("recursive task requires one frozen forest per mechanism")
                return {**raw, "source": forests[0]}
        return {k: _authored(v) for k, v in raw.items()}
    if type(value) in (tuple, list):
        return tuple(_authored(v) for v in _array(value))
    return value


def _domain_width(domain: Any, continuous_width: int) -> int:
    raw = _object(domain)
    if raw.get("kind") == "continuous":
        return continuous_width
    if raw.get("kind") not in ("binary", "categorical"):
        raise ValueError("recursive task has invalid native domain")
    count = len(_array(raw.get("vocabulary")))
    if count < 2:
        raise ValueError("recursive task has invalid category cardinality")
    return count


def admit_recursive_tasks(
    graphs: list[dict[str, Any]],
    *,
    storage: int,
    numbers: int,
    replay_work: int,
    limits: CausalLimits,
) -> None:
    """Conservative shared state, emission, traversal and replay reservation; no RNG access."""
    try:
        for graph in graphs:
            declared = _object(graph.get("limits"))
            if set(declared) != {f.name for f in fields(CausalLimits)}:
                raise ValueError("recursive graph limits require every field")
            effective = recursive_effective_limits(CausalLimits(**declared), limits)
            task_raw = dict(_object(graph.get("task")))
            if "class_offsets" in task_raw:
                task_raw["class_offsets"] = tuple(_array(task_raw["class_offsets"]))
            task = LinearRecipe(**task_raw)
            raw_nodes = _array(graph.get("nodes"))
            if not raw_nodes or len(raw_nodes) > effective.nodes:
                raise ValueError("recursive task node limit exceeded")
            scores: list[int] = []
            encoded: list[int] = []
            values: list[int] = []
            geometry_work, floor, fan_in, root_width, root_total, pipeline_work = 0, 0, 0, 0, 0, 0
            function_work = 0
            members: list[FrozenRecursiveForestFunction] = []
            roots: list[tuple[Any, int]] = []
            converter_work = 0
            parameter_extra = 0
            version = graph["version"]
            for raw in raw_nodes:
                node = _object(raw)
                equation = _object(node.get("equation", node))
                function = decode_admitted_function(
                    _authored(equation.get("function")), limits=effective
                )
                parents = _array(equation.get("parents"))
                if any(type(p) is not int or not 0 <= p < len(encoded) for p in parents) or len(
                    set(parents)
                ) != len(parents):
                    raise ValueError("recursive graph parent geometry must be acyclic and distinct")
                parents = cast(tuple[int, ...], tuple(parents))
                if parents:
                    if equation.get("root") is not None or function.input_width != sum(
                        encoded[p] for p in parents
                    ):
                        raise ValueError("recursive graph encoded parent width differs")
                    if isinstance(function, FrozenParentFunction) and tuple(
                        c.input_width for c in function.children
                    ) != tuple(encoded[p] for p in parents):
                        raise ValueError("recursive graph parent composition slices differ")
                elif (
                    equation.get("root") is None
                    or recursive_root_width(equation["root"]) != function.input_width
                ):
                    raise ValueError("recursive graph root width differs from function input")
                dims, work, minimum_rows = function_geometry(function)
                if max(dims) > effective.node_channels:
                    raise ValueError("recursive task function channel limit exceeded")
                geometry_work = max(geometry_work, work)
                floor = max(floor, minimum_rows)
                fan_in = max(fan_in, function.input_width)
                scores.append(function.output_width)
                if equation.get("root") is not None:
                    root_width = max(root_width, function.input_width)
                    root_total += function.input_width
                    roots.append((equation["root"], function.input_width))
                if (
                    _object(_object(equation.get("exogenous", {})).get("inputs", {})).get("bound")
                    is not None
                ):
                    floor = max(floor, 256)
                pipeline = equation.get("pipeline")
                if pipeline is not None:
                    pipeline_work = max(
                        pipeline_work, functional_pipeline(pipeline).working_extra_channels
                    )
                if version == "native-causal-v1":
                    width = _domain_width(node.get("domain"), function.output_width)
                    encoded.append(width)
                    values.append(
                        function.output_width
                        if _object(node["domain"]).get("kind") == "continuous"
                        else 1
                    )
                elif version == "native-vector-causal-v1":
                    domains = _array(node.get("domains"))
                    encoded.append(sum(_domain_width(d, 1) for d in domains))
                    values.append(len(domains))
                else:
                    encoded.append(function.output_width)
                    values.append(function.output_width)
                members.extend(recursive_function_members(function))
                for converter in _array(node.get("converters", ())):
                    item = _object(converter)
                    width = item["stop"] - item["start"]
                    k = len(_array(item["vocabulary"]))
                    converter_work += 8 * task.rows * k * (3 * width + 8) + 16 * k * width
                    if item.get("post_function") is not None:
                        from r0.modules.priors.lib.frozen_function_evaluation import (
                            admit_frozen_function,
                        )

                        post = decode_admitted_function(
                            _authored(item["post_function"]), limits=effective
                        )
                        report = admit_frozen_function(post, task.rows, effective)
                        converter_work += report["estimated_working_bytes"]
                        parameter_extra += report["parameters"]
            if max(encoded) > effective.node_channels or sum(encoded) > effective.total_channels:
                raise ValueError("recursive task encoded channel limit exceeded")
            calibration = graph.get("calibration_rows", 0)
            if (
                type(calibration) is not int
                or calibration < 0
                or (version == "functional-causal-v2" and not 32 <= calibration <= 65536)
            ):
                raise ValueError("recursive task calibration row geometry differs")
            if len(_array(graph.get("observed"))) != task.features:
                raise ValueError("recursive graph observed width differs from task")
            rows = max(task.rows, floor, calibration)
            for member in members:
                report = member.admit(rows, effective)
                function_work += report["estimated_working_bytes"]
                parameter_extra += report["parameters"]
            for root, width in roots:
                root_numbers, root_work = recursive_root_budget(root, rows, width, effective)
                parameter_extra += root_numbers
                function_work += root_work
            total = sum(scores)
            # Source count is an upper bound even for disconnected/zero-noise nodes.
            source_bound = root_total + total
            work = (
                6 * storage
                + replay_work
                + function_work
                + converter_work
                + 256 * total * source_bound
                + 256 * total
                + 8
                * rows
                * (
                    6 * total
                    + 2 * fan_in
                    + 2 * task.features
                    + 8 * max(1, task.classes, max(encoded))
                    + geometry_work
                    + 40 * root_width
                    + pipeline_work
                    + 4
                )
            )
            if version.startswith("native-"):
                channels = sum(encoded) + 3 * sum(values) + total + len(raw_nodes)
                sources = root_total + total + sum(values) + len(raw_nodes)
                work += 256 * rows * channels + 256 * sum(encoded) * sources
            if work > effective.working_bytes or numbers + parameter_extra > effective.parameters:
                raise ValueError(
                    "recursive whole-task geometry/emission/replay workspace "
                    "or parameter limit exceeded"
                )
    except (KeyError, TypeError, AttributeError, OverflowError, IndexError) as error:
        raise ValueError("malformed recursive whole-task admission geometry") from error
