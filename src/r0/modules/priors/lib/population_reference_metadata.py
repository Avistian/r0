"""Complete reference receipt shapes, semantics and identities before source replay."""

import json
import math
from typing import Any, cast

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.population_reference_admission import admit_population_reference
from r0.modules.priors.lib.representatives import record_fingerprint


def reference_array(value: Any, length: int) -> list[Any] | tuple[Any, ...]:
    if type(value) not in (tuple, list):
        raise ValueError("population reference requires exact array geometry")
    result = cast(list[Any] | tuple[Any, ...], value)
    if len(result) != length:
        raise ValueError("population reference array length differs")
    return result


def validate_reference_metadata(raw: dict[str, Any], limits: CausalLimits) -> dict[str, Any]:
    if (
        raw["version"] != "population-graph-reference-v1"
        or raw["precedence"] != "observed-location-reference-overrides-only-prior-hard-cut-v1"
        or raw["semantics"] != "original-row-owned-mechanisms-and-draws-not-conditional-mean-v1"
        or type(raw["source_json"]) is not str
    ):
        raise ValueError("population reference requires exact versioned semantics")
    source = json.loads(raw["source_json"])
    if type(source) is not dict:
        raise ValueError("population reference requires an object source")
    source = cast(dict[str, Any], source)
    base = source.get("base", source)
    if any(type(raw[k]) is not int or raw[k] != base[k] for k in ("seed", "task_id")):
        raise ValueError("population reference identity differs before replay")
    if raw["source_fingerprint"] != record_fingerprint(source):
        raise ValueError("population reference source fingerprint differs")
    values = raw["values"]
    if type(values) not in (list, tuple) or not 1 <= len(values) <= 129:
        raise ValueError("population reference grid must be a bounded array")
    _, _, schema = admit_population_reference(
        source,
        row=raw["row"],
        feature=raw["feature"],
        values=tuple(values),
        capture_location=raw["capture_location"],
        limits=limits,
    )
    change = source.get("change")
    mode = change["mode"] if change else "original"
    index = change["population"] if change is not None and mode == "population" else None
    if (
        raw["population_mode"] != mode
        or type(raw["population_index"]) is not type(index)
        or raw["population_index"] != index
    ):
        raise ValueError("population reference environment mode/identity differs")
    selected = schema[cast(int, raw["feature"])]
    vocabulary = reference_array(raw["vocabulary"], len(selected.vocabulary))
    if tuple(vocabulary) != selected.vocabulary:
        raise ValueError("population reference vocabulary differs")
    anchor = reference_array(raw["anchor"], len(schema))
    for value, feature in zip(anchor, schema, strict=True):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("population reference anchor must retain finite float64 values")
        feature.decode(value)
    task = base["request"]["base"]["task"]
    classes = task["classes"]
    target_classes = reference_array(raw["target_classes"], classes)
    if (
        any(type(v) is not str or not v for v in target_classes)
        or len(set(target_classes)) != classes
    ):
        raise ValueError("population reference target vocabulary differs")
    scores = reference_array(raw["scores"], len(values))
    width = len(reference_array(scores[0], len(scores[0])))
    if not 1 <= width <= max(1, classes):
        raise ValueError("population reference target score width differs")
    for row in scores:
        for value in reference_array(row, width):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("population reference scores must be finite float64 values")
    if classes:
        for row in reference_array(raw["probabilities"], len(values)):
            probabilities = reference_array(row, classes)
            if (
                any(
                    type(v) is not float or not math.isfinite(v) or not 0 <= v <= 1
                    for v in probabilities
                )
                or abs(sum(probabilities) - 1) > 1e-12
            ):
                raise ValueError("population reference probabilities are invalid")
    elif raw["probabilities"] is not None:
        raise ValueError("regression reference must not advertise class probabilities")
    for target in reference_array(raw["targets"], len(values)):
        if classes:
            if type(target) is not int or not 0 <= target < classes:
                raise ValueError("population reference target category differs")
        elif type(target) is not float or not math.isfinite(target):
            raise ValueError("population reference regression target must be finite float64")
    return source
