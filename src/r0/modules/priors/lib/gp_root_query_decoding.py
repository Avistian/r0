"""Canonical GP-query replay follows complete outer lineage preflight."""

from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_calibration_decoding import gp_root_calibration_from_dict
from r0.modules.priors.lib.gp_root_decoding import gp_root_dag_from_dict
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery
from r0.modules.priors.lib.gp_root_query_metadata import query_source_metadata

_DEFAULT_LIMITS = CausalLimits()


def gp_root_calibration_query_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPRootCalibrationQuery:
    metadata = query_source_metadata(data, limits=limits)
    chain: list[dict[str, Any]] = []
    raw = data
    while raw["version"] == "gp-root-calibration-query-v1":
        chain.append(raw)
        raw = raw["source"]
    if not chain:
        raise ValueError("GP query decoder requires a query wrapper")
    source: GPRootCalibration | GPRootCalibrationQuery = gp_root_calibration_from_dict(
        raw, limits=metadata.limits
    )
    for item in reversed(chain):
        source = GPRootCalibrationQuery(
            source, gp_root_dag_from_dict(item["original"], limits=metadata.limits), metadata.limits
        )
        source.validate()
    if not gp_payload_equal(asdict(source), data):
        raise ValueError("GP query differs from canonical full-lineage replay")
    assert isinstance(source, GPRootCalibrationQuery)
    return source
