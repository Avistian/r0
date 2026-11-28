"""Strict reconstruction of original-row GP causal reference receipts."""

import json
from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import gp_fields, gp_payload_equal
from r0.modules.priors.lib.gp_population_metadata_cost import retained_cost
from r0.modules.priors.lib.gp_reference_admission import admit_gp_reference
from r0.modules.priors.lib.gp_reference_metadata import validate_reference_metadata
from r0.modules.priors.lib.gp_root_reference import GPRootReference, gp_root_reference_from_source
from r0.modules.priors.lib.representatives import encode

_DEFAULT_LIMITS = CausalLimits()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate GP reference source key")
        result[key] = value
    return result


def gp_root_reference_from_dict(
    data: object, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPRootReference:
    if type(limits) is not CausalLimits:
        raise ValueError("GP reference requires explicit caller limits")
    retained_cost(data, limits)
    raw = gp_fields(data, GPRootReference)
    if type(raw["source_json"]) is not str:
        raise ValueError("GP reference source must be canonical JSON")
    try:
        source = json.loads(raw["source_json"], object_pairs_hook=_object)
        if encode(source) != raw["source_json"]:
            raise ValueError("GP reference source must be canonical JSON")
        if type(raw["inputs"]) not in (list, tuple) or any(
            type(point) not in (list, tuple) for point in raw["inputs"]
        ):
            raise ValueError("GP reference grid must be a sequence of rows")
        inputs = tuple(tuple(point) for point in raw["inputs"])
        plan = admit_gp_reference(
            source,
            row=raw["row"],
            inputs=inputs,
            structural_innovations=raw["structural_innovations"],
            capture_location=raw["capture_location"],
            limits=limits,
        )
        validate_reference_metadata(raw, source, plan)
        result = gp_root_reference_from_source(
            source,
            row=raw["row"],
            inputs=inputs,
            structural_innovations=raw["structural_innovations"],
            capture_location=raw["capture_location"],
            limits=limits,
        )
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("malformed GP reference receipt") from error
    if not gp_payload_equal(asdict(result), raw):
        raise ValueError("GP reference receipt differs from retained source replay")
    return result
