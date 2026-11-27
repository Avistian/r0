"""Metadata-first GP population decoding with complete construction replay custody."""

from dataclasses import asdict, fields
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.gp_column_draw import GPColumnKernel
from r0.modules.priors.lib.gp_column_output import GPColumnRecipe
from r0.modules.priors.lib.gp_column_population import (
    GPColumnPopulation,
    GPColumnRequest,
    realize_gp_columns,
)
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.linear import LinearRecipe

_DEFAULT_LIMITS = CausalLimits()


def gp_fields(data: Any, cls: type) -> dict[str, Any]:
    if type(data) is not dict:
        raise ValueError("GP archive must be an object")
    mapping = cast(dict[str, Any], data)
    if set(mapping) != {f.name for f in fields(cls)}:
        raise ValueError("GP archive fields differ from explicit versioned schema")
    version = next((field.default for field in fields(cls) if field.name == "version"), None)
    if version is not None and (
        type(mapping["version"]) is not str or mapping["version"] != version
    ):
        raise ValueError("GP archive version differs from explicit schema")
    return dict(mapping)


def gp_payload_equal(expected: Any, actual: Any) -> bool:
    if type(expected) is dict:
        expected = cast(dict[str, Any], expected)
        actual = cast(dict[str, Any], actual)
        return (
            type(actual) is dict
            and expected.keys() == actual.keys()
            and all(gp_payload_equal(v, actual[k]) for k, v in expected.items())
        )
    if type(expected) in (list, tuple):
        return (
            type(actual) in (list, tuple)
            and len(expected) == len(actual)
            and all(gp_payload_equal(a, b) for a, b in zip(expected, actual, strict=True))
        )
    return type(expected) is type(actual) and expected == actual


def gp_column_request_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPColumnRequest:
    from r0.modules.priors.lib.gp_column_metadata import gp_sequence

    admit_function_metadata(data, limits)
    raw = gp_fields(data, GPColumnRequest)
    gp_sequence(raw["kernels"])
    gp_sequence(raw["locations"])
    for row in raw["locations"]:
        gp_sequence(row)
    raw["kernels"] = tuple(GPColumnKernel(**gp_fields(k, GPColumnKernel)) for k in raw["kernels"])
    raw["locations"] = tuple(tuple(row) for row in raw["locations"])
    raw["limits"] = CausalLimits(**gp_fields(raw["limits"], CausalLimits))
    request = GPColumnRequest(**raw)
    request.admit(limits)
    return request


def gp_column_population_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPColumnPopulation:
    admit_function_metadata(data, limits)
    raw = gp_fields(data, GPColumnPopulation)
    request = gp_population_metadata(raw, limits=limits)
    # The request reserves the whole retained population plus simultaneous replay.
    expected = realize_gp_columns(request, seed=raw["seed"], task_id=raw["task_id"])
    if not gp_payload_equal(asdict(expected), data):
        raise ValueError("GP population differs from full identity-bound replay")
    return expected


def validate_gp_population(population: GPColumnPopulation, *, limits: CausalLimits) -> None:
    if (
        type(population) is not GPColumnPopulation
        or type(population.request) is not GPColumnRequest
    ):
        raise ValueError("GP population requires an exact owned request")
    population.request.admit(limits)
    function_record_cost(population, limits)
    gp_column_population_from_dict(asdict(population), limits=limits)


def gp_column_recipe_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPColumnRecipe:
    admit_function_metadata(data, limits)
    raw = gp_fields(data, GPColumnRecipe)
    source = raw["source"]
    if type(source) is not dict:
        raise ValueError("GP recipe source must be an object")
    source = cast(dict[str, Any], source)
    task = gp_fields(raw["task"], LinearRecipe)
    from r0.modules.priors.lib.gp_column_metadata import gp_sequence

    gp_sequence(task["class_offsets"])
    task["class_offsets"] = tuple(task["class_offsets"])
    raw["task"] = LinearRecipe(**task)
    request = (
        gp_column_request_from_dict(source, limits=limits)
        if source.get("version") == "gp-column-request-r0-v1"
        else gp_population_metadata(source, limits=limits)
    )
    # The request-only recipe validates all outer controls without population replay.
    GPColumnRecipe(request, raw["task"], raw["version"])
    raw["source"] = (
        request
        if source.get("version") == "gp-column-request-r0-v1"
        else gp_column_population_from_dict(source, limits=limits)
    )
    return GPColumnRecipe(**raw)


def gp_population_metadata(data: Any, *, limits: CausalLimits) -> GPColumnRequest:
    """Validate immutable population envelope geometry/identity without numerical replay."""
    from r0.modules.priors.lib.gp_column_metadata import (
        gp_draw_metadata,
        gp_matrix_metadata,
        gp_target_metadata,
        gp_vector_metadata,
    )
    from r0.modules.priors.lib.representatives import record_fingerprint

    raw = gp_fields(data, GPColumnPopulation)
    request = gp_column_request_from_dict(raw["request"], limits=limits)
    if any(type(raw[key]) is not int or not 0 <= raw[key] < 2**63 for key in ("seed", "task_id")):
        raise ValueError("GP population requires exact integer identity")
    expected_id = record_fingerprint((asdict(request), raw["seed"], raw["task_id"]))
    if type(raw["population_id"]) is not str or raw["population_id"] != expected_id:
        raise ValueError("GP population identity differs from replay")
    gp_vector_metadata(raw["row_ids"], request.rows)
    if any(type(v) is not int or v != i for i, v in enumerate(raw["row_ids"])):
        raise ValueError("GP row identity differs from replay")
    gp_draw_metadata(raw["draw"], request)
    target = gp_target_metadata(raw["target"])
    expected_width = request.hidden if request.target_family == "neural" else request.columns
    if target.family != request.target_family or (
        target.input_width,
        target.outputs,
        len(target.weights),
    ) != (request.columns, request.outputs, expected_width):
        raise ValueError("GP population target geometry differs from replay")
    for key in ("calibration_innovations", "calibration_inputs"):
        gp_matrix_metadata(raw[key], request.calibration_rows, request.columns)
    gp_matrix_metadata(raw["calibration_scores"], request.calibration_rows, request.outputs)
    for key in ("raw_scores", "scores"):
        gp_matrix_metadata(raw[key], request.rows, request.outputs)
    for key in ("centers", "scales"):
        gp_vector_metadata(raw[key], request.outputs)
    if any(v <= 0 for v in raw["scales"]):
        raise ValueError("GP population scales differ from replay")
    return request
