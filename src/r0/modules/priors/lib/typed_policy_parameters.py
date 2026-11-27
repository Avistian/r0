"""Declared numerical law realization for a retained typed equation."""

import json
import math
from dataclasses import asdict

import numpy as np

from r0.modules.priors.lib.frozen_function_decoding import frozen_function_from_dict
from r0.modules.priors.lib.intervention_policy_laws import (
    numeric_leaves,
    parameter_group,
    replace_leaves,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw
from r0.modules.priors.lib.typed_policy_request import TypedInterventionPolicyRequest


def resample_typed_function(
    request: TypedInterventionPolicyRequest,
    node: int,
    staged: dict[str, np.random.Generator],
    capture: RawDrawLedger,
):
    original = request.equations[node].function
    record = json.loads(json.dumps(asdict(original), allow_nan=False))
    for index, law in enumerate(request.parameter_laws):
        if law.node != node:
            continue
        parent, key, group = parameter_group(record, law.path)
        size = numeric_leaves(group)
        raw = recorded_draw(
            staged["values"],
            capture,
            owner=("policy", f"node:{node}"),
            role=f"parameter:{index}",
            distribution="normal" if law.family == "normal-r0" else "random",
            shape=(size,),
        )
        if law.family == "normal-r0" or law.family == "uniform-r0":
            values = law.location + law.scale * raw
        elif law.family == "log-uniform-r0":
            values = np.exp((1 - raw) * math.log(law.location) + raw * math.log(law.scale))
        else:
            values = np.asarray(law.choices)[np.floor(raw * len(law.choices)).astype(np.int64)]
        if not np.isfinite(values).all():
            raise ValueError("nonfinite resampled parameter; no retry")
        parent[key] = replace_leaves(group, iter(values))
    function = frozen_function_from_dict(record, limits=request.limits)
    if (
        type(function) is not type(original)
        or function.input_width != original.input_width
        or function.output_width != original.output_width
    ):
        raise ValueError("soft resampling must preserve function family and geometry")
    return function
