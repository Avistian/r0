"""Complete calibration/query lineage semantics and aggregate admission without replay."""

from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.gp_column_decoding import gp_fields, gp_payload_equal
from r0.modules.priors.lib.gp_column_population import GPColumnRequest
from r0.modules.priors.lib.gp_population_metadata_cost import MetadataCost, retained_cost
from r0.modules.priors.lib.gp_root_calibration_decoding import preflight_gp_root_calibration
from r0.modules.priors.lib.gp_root_dags import GPRootBinding, GPRootRuntime
from r0.modules.priors.lib.gp_root_decoding import preflight_gp_root_dag
from r0.modules.priors.lib.gp_root_query import GPRootCalibrationQuery


@dataclass(frozen=True)
class GPQueryMetadata:
    current: dict[str, Any]
    limits: CausalLimits
    base: GPRootRuntime
    bindings: tuple[GPRootBinding, ...]
    request: GPColumnRequest
    identities: frozenset[tuple[int, int]]
    populations: tuple[tuple[GPColumnRequest, GPRootRuntime], ...]


def admit_query_envelope(
    cost: MetadataCost,
    populations: tuple[tuple[GPColumnRequest, GPRootRuntime], ...],
    limits: CausalLimits,
) -> None:
    for ceiling in (limits, *(cap for r, b in populations for cap in (r.limits, b.limits))):
        cost.admit(ceiling)
        reports = [
            report
            for r, b in populations
            for report in (r.admit(ceiling), replace(b, limits=ceiling).admit())
        ]
        if (
            cost.numbers + sum(r["parameters"] for r in reports) > ceiling.parameters
            or 8 * cost.storage + 4 * sum(r["estimated_working_bytes"] for r in reports)
            > ceiling.working_bytes
        ):
            raise ValueError("GP query complete lineage metadata/runtime exceeds limits")


def query_source_metadata(data: Any, *, limits: CausalLimits) -> GPQueryMetadata:
    if type(limits) is not CausalLimits:
        raise ValueError("GP query requires explicit caller limits")
    cost = retained_cost(data, limits)
    try:
        chain: list[dict[str, Any]] = []
        if type(data) is not dict:
            raise ValueError("GP query source must be a complete object")
        root = cast(dict[str, Any], data)
        while root.get("version") == "gp-root-calibration-query-v1":
            raw = gp_fields(root, GPRootCalibrationQuery)
            chain.append(raw)
            if type(raw["source"]) is not dict:
                raise ValueError("GP query lineage requires complete source objects")
            root = cast(dict[str, Any], raw["source"])
        cap, _ = preflight_gp_root_calibration(root, limits=limits)
        current: dict[str, Any] = root["original"]
        _, base, bindings, request = preflight_gp_root_dag(current, limits=cap)
        _, fit_base, _, fit_request = preflight_gp_root_dag(root["fit"], limits=cap)
        identities: set[tuple[int, int]] = {
            (r["source"]["seed"], r["source"]["task_id"]) for r in (current, root["fit"])
        }
        populations = [(request, base), (fit_request, fit_base)]
        for raw in reversed(chain):
            if not gp_payload_equal(raw["limits"], asdict(cap)):
                raise ValueError("GP query must retain original calibration limits")
            next_raw = raw["original"]
            _, next_base, next_bindings, next_request = preflight_gp_root_dag(next_raw, limits=cap)
            if (
                not gp_payload_equal(next_raw["base"], current["base"])
                or not gp_payload_equal(next_raw["bindings"], current["bindings"])
                or not gp_payload_equal(next_raw["limits"], asdict(cap))
                or not gp_payload_equal(
                    asdict(next_request), asdict(replace(request, locations=next_request.locations))
                )
            ):
                raise ValueError("GP query must preserve complete authored graph, bindings and law")
            identity = (next_raw["source"]["seed"], next_raw["source"]["task_id"])
            if identity in identities:
                raise ValueError("GP query identity must differ from every ancestor and fit")
            identities.add(identity)
            populations.append((next_request, next_base))
            current, base, bindings, request = next_raw, next_base, next_bindings, next_request
        admit_query_envelope(cost, tuple(populations), cap)
        return GPQueryMetadata(
            current, cap, base, bindings, request, frozenset(identities), tuple(populations)
        )
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid complete GP query lineage metadata") from exc
