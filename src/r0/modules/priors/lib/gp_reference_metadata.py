"""Receipt semantics and tensor geometry checked before any retained source replay."""

import math
from dataclasses import asdict
from typing import Any

from r0.modules.priors.lib.gp_column_decoding import gp_payload_equal
from r0.modules.priors.lib.gp_reference_admission import GPReferencePlan
from r0.modules.priors.lib.representatives import record_fingerprint


def validate_reference_metadata(
    raw: dict[str, Any], source: dict[str, Any], plan: GPReferencePlan
) -> None:
    expected = {
        "source_fingerprint": record_fingerprint(source),
        "population_id": plan.current["source"]["population_id"],
        "fit_population_id": plan.fit_population_id,
        "seed": plan.current["source"]["seed"],
        "task_id": plan.current["source"]["task_id"],
        "feature_schema": tuple(asdict(f) for f in plan.schema),
        "classes": plan.classes,
        "semantics": "observed-clamps-original-hidden-population-not-conditional-mean-v1",
        "target_draws": "fixed-original-row-task-adapter-and-domain-draws-v1",
    }
    if any(not gp_payload_equal(value, raw[key]) for key, value in expected.items()):
        raise ValueError("GP reference identity, schema or semantics differ from source")
    size = len(raw["inputs"])

    def vector(value: Any, width: int, *, integer: bool = False) -> None:
        if type(value) not in (tuple, list) or len(value) != width:
            raise ValueError("GP reference vector geometry differs")
        for item in value:
            if type(item) is not (int if integer else float) or not math.isfinite(item):
                raise ValueError("GP reference values have invalid primitive type or are nonfinite")
            if integer and not 0 <= item < len(plan.classes):
                raise ValueError("GP reference label outside retained vocabulary")

    def matrix(value: Any, width: int) -> None:
        if type(value) not in (tuple, list) or len(value) != size:
            raise ValueError("GP reference matrix row geometry differs")
        for point in value:
            vector(point, width)

    matrix(raw["raw_scores"], plan.score_width)
    matrix(raw["scores"], plan.score_width)
    vector(raw["targets"], size, integer=bool(plan.classes))
    if plan.classes:
        if raw["target_mean"] is not None:
            raise ValueError("categorical GP references do not have numeric target means")
        matrix(raw["probabilities"], len(plan.classes))
        if any(
            any(not 0 <= p <= 1 for p in row) or abs(sum(row) - 1) > 1e-12
            for row in raw["probabilities"]
        ):
            raise ValueError("GP reference probabilities are invalid")
    else:
        vector(raw["target_mean"], size)
        if raw["probabilities"] is not None:
            raise ValueError("continuous GP reference cannot contain class probabilities")
