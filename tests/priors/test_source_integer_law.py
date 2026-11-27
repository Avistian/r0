"""The source floor/clip rule preserves its degenerate discretization count."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors


def test_source_integer_clip_keeps_normal_and_equal_bound_semantics() -> None:
    rule = "source-clip-floor-v1"
    normal = priors.NumericalParameter(
        "rand_discr_n_centers", 2, 5, shape=(3,), scale="log", integer=rule
    )
    assert priors.NumericalDraw(normal, 0, (0.0, 0.5, 1.0), (2, 3, 4)).values == (2, 3, 4)
    degenerate = priors.NumericalParameter(
        "rand_discr_n_centers", 2, 2, shape=(3,), scale="log", integer=rule
    )
    # The source's clip(floor(2), 2, 1) returns its upper clipping bound, 1.
    assert priors.NumericalDraw(degenerate, 0, (0.0, 0.5, 1.0), (1, 1, 1)).values == (1, 1, 1)
    with pytest.raises(ValueError, match="realization"):
        priors.NumericalDraw(degenerate, 0, (0.0, 0.5, 1.0), (2, 2, 2))
    with pytest.raises(ValueError, match="integer bounds"):
        priors.NumericalParameter("legacy-floor", 2, 2, integer="floor")


def test_source_integer_clip_preserves_named_preferences_and_json_across_row_counts() -> None:
    for mode in ("local", "meta", "global"):
        context = priors.JointSamplingContext(np.random.default_rng(28), mode=mode)
        with context.transaction("function") as operation:
            regular = operation.numerical(
                "regular",
                priors.NumericalParameter(
                    "rand_discr_n_centers", 2, 17, scale="log", integer="source-clip-floor-v1"
                ),
            )
            degenerate = operation.numerical(
                "two-rows",
                priors.NumericalParameter(
                    "rand_discr_n_centers", 2, 2, scale="log", integer="source-clip-floor-v1"
                ),
            )
        assert regular.preference == degenerate.preference
        assert 2 <= regular.values[0] <= 16
        assert degenerate.values == (1,)
        if mode == "global":
            assert regular.units == degenerate.units
        record = context.freeze()
        restored = priors.joint_sampling_from_dict(json.loads(json.dumps(asdict(record))))
        assert restored == record
