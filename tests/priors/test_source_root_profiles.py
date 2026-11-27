"""External NumPy law oracles, frozen parameter lifecycles and strict custody."""

import copy
import json
from dataclasses import asdict, replace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
def test_pinned_conditional_law_oracle_and_replay(engine: str):
    model = np.random.Generator(getattr(np.random, engine)(37))
    selection = np.random.Generator(getattr(np.random, engine)(811))
    m, s = copy.deepcopy(model), copy.deepcopy(selection)
    width = 24
    mu, z = m.normal(size=width), m.normal(size=width)
    dead, multi, normal = tuple(0.66 * float(s.random()) for _ in range(3))
    expected: list[p.SourceRootCoordinate] = []
    roles = [
        "gaussian_means",
        "gaussian_scale_factors",
        "zipf_threshold_unused",
        "categorical_threshold",
        "normal_threshold",
    ]
    for i in range(width):
        roles.append("normal_test")
        if s.random() > normal:
            expected.append(
                p.SourceRootCoordinate(
                    "gaussian", location=float(mu[i]), scale=abs(float(mu[i]) * float(z[i]))
                )
            )
        else:
            roles.append("categorical_test")
            if s.random() > multi:
                k = 2 + int(s.integers(9))
                weights = s.random(k)
                expected.append(
                    p.SourceRootCoordinate(
                        "categorical",
                        probabilities=tuple(float(x) for x in weights / weights.sum()),
                    )
                )
                roles.extend(("category_count", "category_weights"))
            else:
                expected.append(
                    p.SourceRootCoordinate("zipf", exponent=2 + 2 * float(s.random()), clip=10)
                )
                roles.append("zipf_exponent")
    result = p.sample_source_root_profile(
        p.SourceRootProfile(width, presample_gaussian=True),
        model_rng=model,
        selection_rng=selection,
    )
    assert result.law.coordinates == tuple(expected)
    assert result.thresholds == (dead, multi, normal)
    assert [r.role for r in result.raw] == roles
    assert result.profile.source == "tabpfn-mlp-778f0d0-law-selection"
    assert result.profile.lifecycle == "frozen-selection-r0-v1"
    np.testing.assert_array_equal(model.random(7), m.random(7))
    np.testing.assert_array_equal(selection.random(7), s.random(7))
    assert p.sampled_source_root_profile_from_dict(json.loads(json.dumps(asdict(result)))) == result
    # Frozen law emission has no hidden generated-batch normalization.
    output = p.draw_source_root(result.law, rows=13, rng=np.random.default_rng(18))
    for i, law in enumerate(result.law.coordinates):
        if law.family == "categorical":
            assert np.equal(output[:, i], np.floor(output[:, i])).all()
            assert result.source_normalization[i] == "batch-sample-standardization"
        if law.family == "zipf":
            assert (output[:, i] >= 1).all()
            assert (output[:, i] <= 10).all()
            assert result.source_normalization[i] == "batch-centering"


def test_forced_coordinate_variety_and_frozen_gaussian_lifecycle():
    families = (
        "gaussian",
        "presampled_gaussian",
        "unit_uniform",
        "symmetric_uniform",
        "categorical",
        "zipf_clipped",
        "zipf_unbounded",
    )
    profile = p.SourceRootProfile(
        7, selector="forced-r0", families=families, unbounded_exponent=1.7
    )
    result = p.sample_source_root_profile(
        profile, model_rng=np.random.default_rng(12), selection_rng=np.random.default_rng(61)
    )
    assert [c.family for c in result.law.coordinates] == [
        "gaussian",
        "gaussian",
        "unit_uniform",
        "symmetric_uniform",
        "categorical",
        "zipf",
        "zipf",
    ]
    assert result.law.moment_order == pytest.approx(0.7)
    assert result.source_normalization == ("none",) * 7
    assert p.source_root_profile_from_dict(asdict(profile)) == profile
    uniform = p.sample_source_root_profile(
        p.SourceRootProfile(7, selector="p1-code-uniform", presample_gaussian=True),
        model_rng=np.random.default_rng(12),
        selection_rng=np.random.default_rng(61),
    )
    assert len(uniform.model_gaussians) == 7
    assert (
        len(uniform.raw) == 2
    )  # Source constructor draws are retained despite unused law parameters.
    assert all(c.family == "unit_uniform" for c in uniform.law.coordinates)


@pytest.mark.parametrize("probability", [0.0, 0.37, 1.0])
def test_p2_choice_is_once_per_logical_root(probability: float):
    oracle = np.random.default_rng(44)
    expected = "gaussian" if oracle.random() < probability else "symmetric_uniform"
    result = p.sample_source_root_profile(
        p.SourceRootProfile(9, selector="p2-node-mixture-r0", p2_normal_probability=probability),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(44),
    )
    assert {c.family for c in result.law.coordinates} == {expected}
    assert len(result.raw) == 1
    assert result.raw[0].role == "node_family"


@pytest.mark.parametrize(
    "field",
    ["law", "raw", "thresholds", "initial_rng", "final_rng", "lifecycle", "source", "unknown"],
)
def test_strict_profile_replay_rejects_tamper(field: str):
    result = p.sample_source_root_profile(
        p.SourceRootProfile(6, presample_gaussian=True),
        model_rng=np.random.default_rng(3),
        selection_rng=np.random.default_rng(5),
    )
    record = json.loads(json.dumps(asdict(result)))
    if field == "law":
        record["law"]["coordinates"][0]["family"] = "symmetric_uniform"
    elif field == "raw":
        record["raw"][0]["values"][0] += 0.5
    elif field == "thresholds":
        record["thresholds"][0] += 0.01
    elif field in ("initial_rng", "final_rng"):
        record[field][0]["digest"] = "bad"
    elif field == "source":
        record["source"] = "paper-exact"
    elif field == "lifecycle":
        record["profile"]["lifecycle"] = "per-model-source-exact"
    else:
        record["ignored"] = True
    with pytest.raises(ValueError, match="root|snapshot|fields|Gaussian|unused"):
        p.sampled_source_root_profile_from_dict(record)


@pytest.mark.parametrize("kind", ["draws", "numbers", "working", "alias", "late"])
def test_aggregate_caps_and_late_failures_roll_back_both_streams(kind: str):
    profile = p.SourceRootProfile(4, selector="forced-r0", families=("categorical",) * 4)
    if kind == "draws":
        profile = replace(profile, raw_limits=p.RawDrawLimits(draws=1))
    elif kind == "numbers":
        profile = replace(profile, raw_limits=p.RawDrawLimits(numbers=1))
    model, selection = np.random.default_rng(12), np.random.default_rng(13)
    before = copy.deepcopy((model, selection))
    limits = p.CausalLimits(working_bytes=1) if kind == "working" else p.CausalLimits()
    if kind == "late":
        with (
            patch.object(np, "sum", return_value=0.0),
            pytest.raises(ValueError, match="degenerate"),
        ):
            p.sample_source_root_profile(
                profile, model_rng=model, selection_rng=selection, limits=limits
            )
    else:
        with (
            patch.object(
                p.RawDrawLedger, "draw", side_effect=AssertionError("draw before admission")
            ),
            pytest.raises(ValueError, match="limit|distinct"),
        ):
            p.sample_source_root_profile(
                profile,
                model_rng=model,
                selection_rng=model if kind == "alias" else selection,
                limits=limits,
            )
    np.testing.assert_array_equal(model.random(9), before[0].random(9))
    np.testing.assert_array_equal(selection.random(9), before[1].random(9))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width=True),
        dict(width=0),
        dict(width=1025),
        dict(width=2, selector="forced-r0", families=("gaussian",)),
        dict(width=1, selector="paper-multivariate"),
        dict(width=1, p2_normal_probability=0.3),
        dict(width=1, unbounded_exponent=2.1),
    ],
)
def test_invalid_profiles_fail_closed(kwargs: dict[str, Any]):
    with pytest.raises(ValueError, match="root|unused|forced"):
        p.SourceRootProfile(**kwargs)


@pytest.mark.parametrize("selector", ["p1-code-normal", "p1-code-uniform"])
@pytest.mark.parametrize("presampled", [False, True])
def test_source_forced_modes_empty_and_unused_streams(selector: str, presampled: bool):
    model, selection = np.random.default_rng(84), np.random.default_rng(73)
    original = copy.deepcopy(selection)
    result = p.sample_source_root_profile(
        p.SourceRootProfile(3, selector=selector, presample_gaussian=presampled),
        model_rng=model,
        selection_rng=selection,
    )
    assert len(result.raw) == (2 if presampled else 0)
    np.testing.assert_array_equal(selection.random(8), original.random(8))
    assert p.sampled_source_root_profile_from_dict(asdict(result)) == result


@pytest.mark.parametrize("part", ["law", "raw_parameters", "raw_values", "snapshots"])
def test_decoder_caps_metadata_before_replay(part: str):
    result = p.sample_source_root_profile(
        p.SourceRootProfile(1, selector="forced-r0", families=("categorical",)),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(7),
    )
    record = json.loads(json.dumps(asdict(result)))
    if part == "law":
        record["law"]["coordinates"][0]["probabilities"] = [1 / 11] * 11
    elif part == "raw_parameters":
        record["raw"][0]["parameters"] = [["probabilities", [1 / 11] * 11]]
    elif part == "raw_values":
        record["raw"][0]["values"] = [0] * 11
    else:
        record["initial_rng"][0]["payload"] = " " * (result.profile.raw_limits.state_bytes + 1)
    with pytest.raises(ValueError, match="array|bound"):
        p.sampled_source_root_profile_from_dict(record)
