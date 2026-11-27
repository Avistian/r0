"""Frozen fit provenance JSON reconstruction, without rerunning sklearn."""

from typing import Any

from r0.modules.priors.lib.fitted_graph import FittedNodeFit
from r0.modules.priors.lib.fitted_tree_decoding import (
    decode_fitted_tree,
    fitted_array,
    fitted_object,
)
from r0.modules.priors.lib.fitted_tree_fitting import FittedTreeRealization
from r0.modules.priors.lib.fitted_tree_plan import FittedTreePlan
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord


def decode_fitted_raw_draw(data: Any) -> RawDrawRecord:
    values = fitted_object(data, RawDrawRecord)
    for key in ("owner", "shape", "values"):
        values[key] = fitted_array(values.get(key), 1048576)
    pairs = tuple(fitted_array(p, 2) for p in fitted_array(values.get("parameters"), 32))
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("fitted raw parameters require name/value pairs")
    values["parameters"] = tuple(
        (pair[0], fitted_array(pair[1], 1048576) if type(pair[1]) is list else pair[1])
        for pair in pairs
    )
    return RawDrawRecord(**values)


def decode_node_fit(data: Any) -> FittedNodeFit:
    node = fitted_object(data, FittedNodeFit)
    fit = fitted_object(node.get("realization"), FittedTreeRealization)
    fit["plan"] = FittedTreePlan(**fitted_object(fit.get("plan"), FittedTreePlan))
    fit["function"] = decode_fitted_tree(fit["function"])
    for key in ("inputs", "targets"):
        fit[key] = tuple(fitted_array(row, 256) for row in fitted_array(fit.get(key), 1024))
    for key in ("backend_seeds", "backend_parameters"):
        fit[key] = fitted_array(fit.get(key), 16)
    for key in ("initial_rng", "final_rng"):
        fit[key] = tuple(
            GeneratorSnapshot(**fitted_object(v, GeneratorSnapshot))
            for v in fitted_array(fit.get(key), 3)
        )
    fit["raw_draws"] = tuple(
        decode_fitted_raw_draw(v) for v in fitted_array(fit.get("raw_draws"), 3)
    )
    node["realization"] = FittedTreeRealization(**fit)
    for key in ("prefix_initial", "prefix_final"):
        node[key] = GeneratorSnapshot(**fitted_object(node.get(key), GeneratorSnapshot))
    node["prefix_draws"] = tuple(
        decode_fitted_raw_draw(v) for v in fitted_array(node.get("prefix_draws"), 65536)
    )
    return FittedNodeFit(**node)
