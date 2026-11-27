"""Bounded sampled graph reopening by complete construction replay and comparison."""

from dataclasses import fields, is_dataclass
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exogenous import ExogenousDistribution, NoiseResponse
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.functional_causal_decoding import (
    functional_array,
    functional_pipeline,
    functional_record,
)
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.joint_function_prior import JointFunctionPrior
from r0.modules.priors.lib.joint_graph_request import JointGraphNoise, JointGraphRequest
from r0.modules.priors.lib.joint_graph_sampling import sample_joint_graph
from r0.modules.priors.lib.joint_matrix_plans import JointMatrixPrior
from r0.modules.priors.lib.joint_root_plans import JointRootPrior
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph

_DEFAULT_LIMITS = CausalLimits()


def _function_prior(raw: Any) -> JointFunctionPrior:
    value = functional_record(raw, JointFunctionPrior)
    matrix = functional_record(value.get("matrix", {}), JointMatrixPrior)
    for name in ("families", "simple_families"):
        if name in matrix:
            matrix[name] = functional_array(matrix[name])
    value["matrix"] = JointMatrixPrior(**matrix)
    return JointFunctionPrior(**value)


def _request(raw: Any, limits: CausalLimits) -> JointGraphRequest:
    value = functional_record(raw, JointGraphRequest)
    declared = CausalLimits(**functional_record(value.get("limits", {}), CausalLimits))
    # Do not rewrite a retained request or its admission record. Larger declared
    # caps require explicit caller authority even if this particular graph is small.
    if any(getattr(declared, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
        raise ValueError("sampled graph declared limits exceed caller limits")
    value["limits"] = declared
    task = functional_record(value["task"], LinearRecipe)
    if "class_offsets" in task:
        task["class_offsets"] = functional_array(task["class_offsets"])
    value["task"] = LinearRecipe(**task)
    for name in ("parents", "observed", "target"):
        value[name] = tuple(functional_array(row) for row in functional_array(value[name]))
    value["widths"] = functional_array(value["widths"])
    value["function_prior"] = _function_prior(value["function_prior"])
    root = functional_record(value["root_prior"], JointRootPrior)
    root["families"] = functional_array(root["families"])
    root["function"] = _function_prior(root["function"])
    value["root_prior"] = JointRootPrior(**root)
    value["raw_limits"] = RawDrawLimits(**functional_record(value["raw_limits"], RawDrawLimits))
    noises: list[JointGraphNoise] = []
    for raw_noise in functional_array(value.get("noise", ())):
        noise = functional_record(raw_noise, JointGraphNoise)
        noise["scales"] = functional_array(noise.get("scales", ()))
        exogenous = functional_record(noise.get("exogenous", {}), ExogenousDistribution)
        exogenous["inputs"] = InputDistribution(
            **functional_record(exogenous.get("inputs", {}), InputDistribution)
        )
        exogenous["factor"] = tuple(
            functional_array(row) for row in functional_array(exogenous.get("factor", ()))
        )
        noise["exogenous"] = ExogenousDistribution(**exogenous)
        noise["response"] = NoiseResponse(
            **functional_record(noise.get("response", {}), NoiseResponse)
        )
        noises.append(JointGraphNoise(**noise))
    value["noise"] = tuple(noises)
    value["authored_pipelines"] = tuple(
        None if p is None else functional_pipeline(p)
        for p in functional_array(value.get("authored_pipelines", ()))
    )
    return JointGraphRequest(**value)


def sampled_graph_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> SampledFunctionalGraph:
    """Reject altered retained fields; do not repair payloads using their seeds."""
    if type(limits) is not CausalLimits:
        raise ValueError("sampled graph decoding requires caller-owned limits")
    admit_function_metadata(data, limits)
    try:
        record = functional_record(data, SampledFunctionalGraph)
        if record.get("version") != "sampled-functional-graph-v1":
            raise ValueError("unsupported sampled graph version")
        request = _request(record["request"], limits)
        admit_function_metadata(data, request.limits)
        initial = functional_array(record["initial"])
        if len(initial) != 4:
            raise ValueError("sampled graph requires four initial stream snapshots")
        snapshots = tuple(
            GeneratorSnapshot(**functional_record(s, GeneratorSnapshot)) for s in initial
        )
        streams = tuple(s.restore(byte_limit=request.raw_limits.state_bytes) for s in snapshots)
        graph = sample_joint_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        if not _matches(graph, data):
            raise ValueError("sampled graph archive differs from complete replay")
        return graph
    except (TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid sampled graph archive") from exc


def _matches(expected: Any, actual: Any) -> bool:
    """Compare admitted JSON directly, without duplicating the graph as dict/text."""
    if is_dataclass(expected) and not isinstance(expected, type):
        names = tuple(f.name for f in fields(expected))
        if type(actual) is not dict:
            return False
        mapping = cast(dict[str, Any], actual)
        return set(mapping) == set(names) and all(
            _matches(getattr(expected, name), mapping[name]) for name in names
        )
    if type(expected) is tuple:
        if type(actual) not in (tuple, list):
            return False
        left_values = cast(tuple[Any, ...], expected)
        right_values = cast(tuple[Any, ...] | list[Any], actual)
        return len(left_values) == len(right_values) and all(
            _matches(left, right) for left, right in zip(left_values, right_values, strict=True)
        )
    return type(expected) is type(actual) and expected == actual
