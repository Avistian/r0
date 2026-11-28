"""Strict GP intervention decoding binds controls and full populations before replay."""

from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import (
    gp_column_population_from_dict,
    gp_fields,
    gp_payload_equal,
    gp_population_metadata,
)
from r0.modules.priors.lib.gp_population_metadata_cost import retained_cost
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_dags import GPRootDAGRecipe
from r0.modules.priors.lib.gp_root_intervention import GPRootIntervention, GPRootInterventionSource
from r0.modules.priors.lib.gp_root_intervention_metadata import intervention_plan
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery
from r0.modules.priors.lib.gp_root_query_metadata import admit_query_envelope


def decode_gp_intervention_source(data: Any, *, limits: CausalLimits) -> GPRootInterventionSource:
    from r0.modules.priors.lib.generate import recipe_from_dict

    source = recipe_from_dict(data, limits=limits)
    if not isinstance(source, (GPRootDAGRecipe, GPRootCalibration, GPRootCalibrationQuery)):
        raise ValueError("GP intervention source version differs")
    return source


def preflight_gp_intervention(data: Any, *, limits: CausalLimits):
    if type(limits) is not CausalLimits:
        raise ValueError("GP intervention requires explicit caller limits")
    cost = retained_cost(data, limits)
    try:
        raw = gp_fields(data, GPRootIntervention)
        metadata, preview, seed, task_id = intervention_plan(
            raw["base"], raw["change"], limits=limits
        )
        if not gp_payload_equal(raw["limits"], asdict(metadata.limits)):
            raise ValueError("GP intervention must retain original limits")
        request = gp_population_metadata(raw["population"], limits=metadata.limits)
        if (
            not gp_payload_equal(asdict(request), asdict(metadata.request))
            or any(type(raw["population"][k]) is not int for k in ("seed", "task_id"))
            or (raw["population"]["seed"], raw["population"]["task_id"]) != (seed, task_id)
        ):
            raise ValueError("GP intervention population law or exact identity differs")
        if preview.mode == "matched" and not gp_payload_equal(
            raw["population"], metadata.current["source"]
        ):
            raise ValueError("matched GP intervention must retain its complete original draw")
        admit_query_envelope(
            cost, metadata.populations + ((request, preview.effective_runtime()),), metadata.limits
        )
        return raw, metadata
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed GP intervention archive") from exc


def gp_root_intervention_from_dict(data: Any, *, limits: CausalLimits) -> GPRootIntervention:
    raw, metadata = preflight_gp_intervention(data, limits=limits)
    result = GPRootIntervention(
        decode_gp_intervention_source(raw["base"], limits=metadata.limits),
        intervention_from_dict(raw["change"], limits=metadata.limits),
        gp_column_population_from_dict(raw["population"], limits=metadata.limits),
        metadata.limits,
    )
    if not gp_payload_equal(asdict(result), data):
        raise ValueError("GP intervention differs from complete canonical replay")
    return result
