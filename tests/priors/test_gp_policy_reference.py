"""GP policy extrema must use retained GP roots, never ordinary replacement root draws."""

from dataclasses import asdict

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p
from r0.modules.priors.lib.gp_policy_reference import realize_gp_policy_reference


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize("lifecycle", [0, 1, 2])
@pytest.mark.parametrize("bounds", ["source-reference-population", "independent-calibration-r0"])
def test_gp_reference_literal_roots(family: str, lifecycle: int, bounds: str):
    source = gp_root_recipe(family)
    if lifecycle:
        source = p.calibrate_gp_root_dag(
            source, calibration_rows=32, calibration_seed=61, calibration_task_id=67
        )
    if lifecycle == 2:
        assert isinstance(source, p.GPRootCalibration)
        source = p.realize_gp_root_calibration_query(source, seed=71, task_id=73)
    current = source if isinstance(source, p.GPRootDAGRecipe) else source.original
    address = (
        p.InterventionPolicyLocation(0, "native-variable", variable=0, location="native-value")
        if family in ("native", "vector")
        else p.InterventionPolicyLocation(0, channels=(0, 1))
    )
    controls = p.TypedInterventionPolicyRequest(
        current.base,
        (1,),
        (address,),
        eligibility=(0,),
        count=1,
        reference_rows=16 if bounds == "source-reference-population" else 9,
        reference_seed=current.source.seed,
        reference_task_id=current.source.task_id,
        bounds_lifecycle=bounds,
    )
    result = realize_gp_policy_reference(
        source,
        controls,
        seed=current.source.seed if bounds == "source-reference-population" else 127,
    )
    assert asdict(result.source) == asdict(source)
    actual = np.asarray(result.values[0])
    expected = np.asarray(result.population.draw.values)
    if family == "vector":
        expected = expected[:, :1]
    np.testing.assert_array_equal(actual, expected)
    if bounds == "source-reference-population":
        assert asdict(result.population) == asdict(current.source)
    else:
        assert result.population.population_id != current.source.population_id
        assert result.population.request.rows == 9
    assert all(record.role != "root-base" for record in result.raw)


@pytest.mark.parametrize("case", ["source-identity", "independent-identity"])
def test_reference_identity_rejects_before_replay(case: str, monkeypatch: pytest.MonkeyPatch):
    from typing import Any

    from r0.modules.priors.lib import gp_policy_reference as module

    source = gp_root_recipe("functional")
    controls = p.TypedInterventionPolicyRequest(
        source.base,
        (1,),
        (p.InterventionPolicyLocation(0, channels=(0, 1)),),
        eligibility=(0,),
        count=1,
        reference_rows=16,
        reference_seed=47,
        reference_task_id=59,
        bounds_lifecycle="source-reference-population"
        if case == "source-identity"
        else "independent-calibration-r0",
    )

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("source replay preceded identity rejection")

    monkeypatch.setattr(module, "decode_gp_intervention_source", forbidden)
    monkeypatch.setattr(module, "realize_gp_columns", forbidden)
    with pytest.raises(ValueError, match="identity"):
        realize_gp_policy_reference(source, controls, seed=48 if case == "source-identity" else 47)
