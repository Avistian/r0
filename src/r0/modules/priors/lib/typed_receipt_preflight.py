"""Complete typed receipt metadata admission before retained source replay."""

import math
from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.causal_intervention_decoding import intervention_from_dict
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fitted_fit_decoding import decode_fitted_raw_draw
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot
from r0.modules.priors.lib.intervention_policy_decoding import complete_record
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.representatives import encode
from r0.modules.priors.lib.typed_policy_preview import typed_authored_preview
from r0.modules.priors.lib.typed_policy_records import TypedInterventionBounds
from r0.modules.priors.lib.typed_policy_request_decoding import typed_request_metadata


def array(value: Any) -> tuple[Any, ...]:
    if type(value) not in (tuple, list):
        raise ValueError("typed receipt requires explicit arrays")
    return tuple(value)


def preflight_typed_receipt(raw: dict[str, Any], *, limits: CausalLimits) -> None:
    if raw["version"] != "sampled-typed-intervention-policy-v1":
        raise ValueError("typed receipt version differs")
    request, graph, effective = typed_request_metadata(raw["request"], limits=limits)
    raw_limits: RawDrawLimits = request["raw_limits"]
    snapshots: list[tuple[GeneratorSnapshot, ...]] = []
    for key in ("initial", "final"):
        values = array(raw[key])
        if len(values) != 3:
            raise ValueError("typed policy requires three initial and final snapshots")
        items = tuple(GeneratorSnapshot(**complete_record(s, GeneratorSnapshot)) for s in values)
        for item in items:
            item.validate_metadata(byte_limit=raw_limits.state_bytes)
        snapshots.append(items)
    initial, final = snapshots
    if any(a.engine != b.engine for a, b in zip(initial, final, strict=True)):
        raise ValueError("typed snapshot engine changes across construction")
    all_records: dict[str, tuple[RawDrawRecord, ...]] = {}
    for group in ("raw", "reference_raw"):
        records = tuple(decode_fitted_raw_draw(item) for item in array(raw[group]))
        all_records[group] = records
        tails: dict[str, str] = {}
        for index, record in enumerate(records):
            if record.sequence != index:
                raise ValueError("typed raw sequence differs")
            if record.stream in tails and tails[record.stream] != record.state_before:
                raise ValueError("typed raw state chain differs")
            tails[record.stream] = record.state_after
        if group == "raw":
            names = ("selection", "values", "calibration")
            if set(tails) - set(names):
                raise ValueError("typed raw stream is unknown")
            for index, name in enumerate(names):
                owned = [record for record in records if record.stream == name]
                if owned and (
                    owned[0].state_before != initial[index].digest
                    or any(record.generator != initial[index].engine for record in owned)
                ):
                    raise ValueError("typed raw construction initial state differs")
                if tails.get(name, initial[index].digest) != final[index].digest:
                    raise ValueError("typed raw construction final state differs")
    selected = array(raw["selected"])
    eligible = request["eligibility"] or tuple(
        i for i in range(len(graph.nodes)) if i not in request["label_nodes"]
    )
    count = request["count"] or math.ceil(request["fraction"] * len(eligible))
    if (
        len(selected) != count
        or any(type(i) is not int or i not in eligible for i in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("typed selected addresses differ from eligible count")
    draws = all_records["raw"]
    if (
        not draws
        or draws[0].owner != ("typed-policy",)
        or draws[0].role != "eligible-permutation"
        or draws[0].stream != "selection"
        or draws[0].distribution != "permutation"
        or draws[0].shape != (len(eligible),)
    ):
        raise ValueError("typed selection draw metadata differs")
    order = draws[0].values
    if (
        set(order) != set(range(len(eligible)))
        or tuple(eligible[int(i)] for i in order[:count]) != selected
    ):
        raise ValueError("typed selected nodes differ from retained permutation")
    seed = raw["reference_seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("typed reference identity must be an exact bounded integer")
    numeric = {
        i: len(address.channels) or 1
        for i, address in enumerate(request["locations"])
        if not address.vocabulary(graph.nodes[address.node])
    }
    if (
        request["kind"] != "hard"
        or not numeric
        or request["bounds_lifecycle"] == "source-reference-population"
    ) and seed != request["reference_seed"]:
        raise ValueError("typed reference identity differs from declared lifecycle")
    calibration = [d for d in draws if d.stream == "calibration"]
    independent = (
        request["kind"] == "hard"
        and bool(numeric)
        and request["bounds_lifecycle"] == "independent-calibration-r0"
    )
    if independent:
        if (
            len(calibration) != 1
            or calibration[0].owner != ("typed-policy",)
            or calibration[0].role != "calibration-seed"
            or calibration[0].distribution != "integers"
            or calibration[0].shape != (1,)
            or calibration[0].parameters != (("high", 2**63 - 1),)
            or calibration[0].values != (seed,)
        ):
            raise ValueError("typed reference seed differs from calibration draw")
    elif calibration:
        raise ValueError("typed lifecycle must not contain calibration draws")
    if (request["kind"] != "hard" or not numeric) and all_records["reference_raw"]:
        raise ValueError("typed policy must not contain unused reference draws")
    expected = [i for i in numeric if request["locations"][i].node in selected]
    bounds = array(raw["bounds"])
    if len(bounds) != len(expected):
        raise ValueError("typed bounds cardinality differs")
    for data, index in zip(bounds, expected, strict=True):
        bound = complete_record(data, TypedInterventionBounds)
        lo, hi = array(bound["minimum"]), array(bound["maximum"])
        if (
            bound["version"] != "typed-intervention-bounds-v1"
            or type(bound["location"]) is not int
            or bound["location"] != index
            or len(lo) != numeric[index]
            or len(hi) != numeric[index]
            or any(type(v) is not float or not math.isfinite(v) for v in (*lo, *hi))
            or any(a > b for a, b in zip(lo, hi, strict=True))
        ):
            raise ValueError("typed bound location, geometry or finite ordering differs")
    runtime = complete_record(raw["runtime"], CausalInterventionRecipe)
    if (
        encode(runtime["base"]) != encode(raw["request"]["base"])
        or runtime["mode"] != request["mode"]
        or runtime["population"] != request["population"]
        or type(runtime["population"]) is not int
    ):
        raise ValueError("invalid typed runtime original base or comparison identity")
    preview = typed_authored_preview(runtime)
    preview["base"] = asdict(graph)
    decoded = intervention_from_dict(preview, limits=effective)
    expected_nodes = (
        [a.node for a in request["locations"] if a.node in selected]
        if request["kind"] == "hard"
        else list(selected)
    )
    if [c.node for c in decoded.interventions] != expected_nodes:
        raise ValueError("typed runtime selected addresses differ")
    if request["kind"] == "hard":
        addresses = [a for a in request["locations"] if a.node in selected]
        for address, change in zip(addresses, decoded.interventions, strict=True):
            values = getattr(change, "values", ())
            value = getattr(change, "value", None)
            if value is not None:
                values = (value,)
            location = request["locations"].index(address)
            if location in expected:
                bound = bounds[expected.index(location)]
                # Convex interpolation may round outside an equal endpoint by an ulp.
                # Exact raw-quantile arithmetic is still checked by construction replay.
                if len(values) != numeric[location] or any(
                    not lo - 4 * max(math.ulp(lo), math.ulp(hi))
                    <= v
                    <= hi + 4 * max(math.ulp(lo), math.ulp(hi))
                    for v, lo, hi in zip(values, bound["minimum"], bound["maximum"], strict=True)
                ):
                    raise ValueError("typed runtime clamp lies outside retained bounds")
            expected_change = address.clamp(
                graph.nodes[address.node], values, getattr(change, "category", None)
            )
            if encode(asdict(expected_change)) != encode(asdict(change)):
                raise ValueError("typed runtime address differs from selected request location")
    else:
        from r0.modules.priors.lib.causal_intervention_records import SoftIntervention

        if any(
            type(c) is not SoftIntervention
            or c.function is None
            or c.noise_scale is not None
            or c.noise_response is not None
            or c.exogenous is not None
            for c in decoded.interventions
        ):
            raise ValueError("typed soft runtime requires only selected function changes")
