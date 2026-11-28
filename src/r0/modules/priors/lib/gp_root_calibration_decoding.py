"""Admit both retained GP-root populations and their fit linkage before replay."""

from dataclasses import asdict, fields, replace
from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_target_calibration import target_calibration_from_dict
from r0.modules.priors.lib.gp_column_decoding import gp_fields, gp_payload_equal
from r0.modules.priors.lib.gp_root_calibration import GPRootCalibration
from r0.modules.priors.lib.gp_root_decoding import gp_root_dag_from_dict, preflight_gp_root_dag
from r0.modules.priors.lib.observation_envelope_admission import observation_metadata_cost
from r0.modules.priors.lib.representatives import record_fingerprint

_DEFAULT_LIMITS = CausalLimits()


def preflight_gp_root_calibration(data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS):
    observation_metadata_cost(data, limits)
    try:
        raw = gp_fields(data, GPRootCalibration)
        if raw["version"] != "gp-root-calibration-v1":
            raise ValueError("unsupported GP calibration version")
        cap = CausalLimits(**gp_fields(raw["limits"], CausalLimits))
        if any(getattr(cap, f.name) > getattr(limits, f.name) for f in fields(CausalLimits)):
            raise ValueError("GP calibration declared limits exceed caller limits")
        _, original, bindings, request = preflight_gp_root_dag(raw["original"], limits=cap)
        _, fit, fit_bindings, fit_request = preflight_gp_root_dag(raw["fit"], limits=cap)
        width = len(original.target) if isinstance(original, FunctionalCausalRecipe) else 1
        calibration = target_calibration_from_dict(raw["calibration"], width)
        if calibration.prefix_fingerprint != record_fingerprint(raw["original"]):
            raise ValueError("GP calibration prefix fingerprint differs from retained source")
        if not 32 <= calibration.rows <= 1024:
            raise ValueError("GP calibration rows exceed finite population geometry")
        if (
            not isinstance(original, FunctionalCausalRecipe)
            and original.task.task_type != "regression"
        ):
            raise ValueError("direct GP category targets require category-preserving calibration")
        expected_base = asdict(original)
        expected_base["task"] = asdict(
            replace(original.task, rows=calibration.rows, context_rows=calibration.rows // 2)
        )
        # Compare the complete raw graph as well as its deterministic semantic preview:
        # the preview deliberately omits stochastic reconstruction evidence.
        raw_expected = dict(raw["original"]["base"])
        raw_expected["task"] = dict(raw_expected["task"])
        raw_expected["task"].update(rows=calibration.rows, context_rows=calibration.rows // 2)
        expected_request = replace(request, rows=calibration.rows, locations=fit_request.locations)
        if (
            not gp_payload_equal(raw_expected, raw["fit"]["base"])
            or not gp_payload_equal(expected_base, asdict(fit))
            or not gp_payload_equal(asdict(expected_request), asdict(fit_request))
            or bindings != fit_bindings
        ):
            raise ValueError("GP calibration graph or source law differs from original")
        identities = [
            (raw[name]["source"]["seed"], raw[name]["source"]["task_id"])
            for name in ("original", "fit")
        ]
        if identities[0] == identities[1]:
            raise ValueError("GP calibration requires an independent population identity")
        for ceiling in (cap, request.limits, fit_request.limits, original.limits, fit.limits):
            count, storage = observation_metadata_cost(data, ceiling)
            reports = [r.admit(ceiling) for r in (request, fit_request)]
            reports += [replace(r, limits=ceiling).admit() for r in (original, fit)]
            if (
                count + sum(r["parameters"] for r in reports) > ceiling.parameters
                or 8 * storage + 4 * sum(r["estimated_working_bytes"] for r in reports)
                > ceiling.working_bytes
            ):
                raise ValueError("GP calibration aggregate source metadata exceeds limits")
        return cap, calibration
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("malformed GP root calibration archive") from exc


def gp_root_calibration_from_dict(data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS):
    cap, calibration = preflight_gp_root_calibration(data, limits=limits)
    raw = gp_fields(data, GPRootCalibration)
    result = GPRootCalibration(
        gp_root_dag_from_dict(raw["original"], limits=cap),
        gp_root_dag_from_dict(raw["fit"], limits=cap),
        calibration,
        cap,
        raw["version"],
    )
    result.validate()
    if not gp_payload_equal(asdict(result), data):
        raise ValueError("GP root calibration differs from canonical replay")
    return result
