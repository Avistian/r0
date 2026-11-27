"""Sampled roots reject false-valued objects in optional custody fields."""

import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def root(fitted: bool) -> p.SampledSourceRootBase:
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="p1-code-normal"),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(2),
    )
    return p.sample_source_root_base(
        profile,
        modes=("standardize" if fitted else "identity",),
        calibration_rows=16 if fitted else 0,
        calibration_rng=np.random.default_rng(3) if fitted else None,
    )


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("field", ["calibration", "calibration_initial", "calibration_final"])
@pytest.mark.parametrize("value", [False, 0, 0.0, "", (), [], {}])
def test_non_none_false_values_are_not_absent_custody(fitted: bool, field: str, value: Any):
    with pytest.raises(ValueError, match="root|calibration|custody|metadata"):
        replace(root(fitted), **{field: value})


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("value", [False, 0, 0.0, "", [], {}, None])
def test_raw_custody_requires_the_frozen_tuple(fitted: bool, value: Any):
    with pytest.raises(ValueError, match="provenance"):
        replace(root(fitted), calibration_raw=value)


@pytest.mark.parametrize("fitted", [False, True])
def test_inherited_calibration_law_guard_precedes_replay(
    fitted: bool, monkeypatch: pytest.MonkeyPatch
):
    valid = root(fitted)
    wrong = p.SourceRootCalibration(
        p.SourceRootLaw((p.SourceRootCoordinate("unit_uniform"),)),
        ("identity",),
        16,
        (0.0,),
        (1.0,),
    )

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("known mismatched inherited field reached source replay")

    monkeypatch.setattr(p.SampledSourceRootProfile, "validate_replay", forbidden)
    with pytest.raises(ValueError, match="root|calibration"):
        replace(valid, calibration=wrong)


@pytest.mark.parametrize("fitted", [False, True])
def test_valid_envelopes_still_generate_reopen_and_preserve_exact_record(fitted: bool):
    valid = root(fitted)
    recipe = p.NetworkRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        p.FrozenNetworkFunction(1, (p.NetworkBlock((0,), (p.NetworkAffine(((1.0,),), (0.0,)),)),)),
        valid,
    )
    before = p.generate_task(recipe, seed=3, task_id=7)
    decoded = p.recipe_from_dict(asdict(recipe))
    after = p.generate_task(decoded, seed=3, task_id=7)
    assert before.recipe_json == after.recipe_json
    assert torch.equal(before.inputs, after.inputs)
    assert torch.equal(before.targets, after.targets)
    assert json.loads(before.recipe_json)["recipe"]["root"] == json.loads(json.dumps(asdict(valid)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calibration", None),
        ("calibration_initial", None),
        ("calibration_final", None),
        ("calibration_raw", None),
        ("calibration_raw", ()),
    ],
)
def test_fitted_envelope_cannot_omit_required_custody(field: str, value: Any):
    with pytest.raises(ValueError, match="provenance|custody"):
        replace(root(True), **{field: value})
