"""Sampled provenance uses exact types; authored transforms keep numeric compatibility."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from r0.modules import priors as p


def selected() -> p.SampledSourceRootProfile:
    return p.sample_source_root_profile(
        p.SourceRootProfile(2, selector="p1-code-uniform", presample_gaussian=True),
        model_rng=np.random.default_rng(31),
        selection_rng=np.random.default_rng(32),
    )


def root() -> p.SampledSourceRootBase:
    return p.sample_source_root_base(
        selected(),
        modes=("center", "identity"),
        calibration_rows=5,
        calibration_rng=np.random.default_rng(33),
    )


@pytest.mark.parametrize(("field", "index"), [("locations", 1), ("scales", 0), ("scales", 1)])
def test_sampled_calibration_rejects_equal_integer_substitution(field: str, index: int):
    valid = root()
    assert valid.calibration is not None
    values = list(getattr(valid.calibration, field))
    assert type(values[index]) is float
    values[index] = int(values[index])
    changed = replace(valid.calibration, **{field: tuple(values)})
    # The authored record stays legitimate; only its sampled-custody claim is false.
    assert p.source_root_calibration_from_dict(asdict(changed)) == changed
    with pytest.raises(ValueError, match="replay|provenance"):
        replace(valid, calibration=changed)


@pytest.mark.parametrize("field", ["location", "scale", "exponent"])
@pytest.mark.parametrize("scope", ["root", "calibration", "profile"])
def test_derived_law_types_must_match_exactly(field: str, scope: str):
    valid = root()
    coordinate = valid.law.coordinates[0]
    changed = replace(coordinate, **{field: int(getattr(coordinate, field))})
    law = replace(valid.law, coordinates=(changed, valid.law.coordinates[1]))
    assert p.source_root_base_from_dict(asdict(p.SourceRootBase(law))).law == law

    def mutate():
        if scope == "root":
            replace(valid, law=law)
        elif scope == "calibration":
            assert valid.calibration is not None
            replace(valid, calibration=replace(valid.calibration, law=law))
        else:
            replace(valid.profile, law=law)

    with pytest.raises(ValueError, match="replay|provenance"):
        mutate()


def test_unused_model_parameters_remain_exact_custody():
    profile = selected()
    gaussian = replace(profile.model_gaussians[0], exponent=2)
    with pytest.raises(ValueError, match="replay"):
        replace(profile, model_gaussians=(gaussian, profile.model_gaussians[1]))


def test_raw_law_parameter_types_are_part_of_calibration_custody():
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(
            1, selector="forced-r0", families=("zipf_unbounded",), unbounded_exponent=3.0
        ),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(2),
    )
    valid = p.sample_source_root_base(
        profile, modes=("center",), calibration_rows=8, calibration_rng=np.random.default_rng(3)
    )
    raw = replace(valid.calibration_raw[0], parameters=(("exponent", 3),))
    assert raw == valid.calibration_raw[0]  # Python equality hides the changed type.
    with pytest.raises(ValueError, match="replay"):
        replace(valid, calibration_raw=(raw,))


@pytest.mark.parametrize("exponent", [3, 3.0])
def test_explicit_request_numeric_types_are_preserved_when_replayed(exponent: int | float):
    profile = p.sample_source_root_profile(
        p.SourceRootProfile(
            1, selector="forced-r0", families=("zipf_unbounded",), unbounded_exponent=exponent
        ),
        model_rng=np.random.default_rng(1),
        selection_rng=np.random.default_rng(2),
    )
    valid = p.sample_source_root_base(
        profile, modes=("center",), calibration_rows=8, calibration_rng=np.random.default_rng(3)
    )
    decoded = p.sampled_source_root_base_from_dict(asdict(valid))
    assert type(decoded.law.coordinates[0].exponent) is type(exponent)
    assert decoded == valid
