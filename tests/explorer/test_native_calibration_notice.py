"""Typed view metadata distinguishes fixed target calibration from raw targets."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.explorer import inspect_task


@pytest.mark.parametrize("family", ["native", "vector", "converted"])
def test_typed_view_retains_independent_target_calibration(family: str):
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=2, signal_features=2, active_features=2
    )
    root = p.FunctionalCausalNode(
        (), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))), p.JointRootBase("gaussian", 2)
    )
    child = p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, -1.0),)))
    if family == "native":
        base = p.NativeCausalRecipe(
            controls,
            (p.NativeCausalNode(root), p.NativeCausalNode(child)),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    elif family == "vector":
        base = p.NativeVectorRecipe(
            controls,
            (
                p.NativeVectorNode(root, (p.VariableDomain(),) * 2),
                p.NativeVectorNode(child, (p.VariableDomain(),)),
            ),
            ((0, 0), (0, 1)),
            (1, 0),
        )
    else:
        base = p.ConvertedCausalRecipe(
            controls,
            (p.ConvertedCausalNode(root), p.ConvertedCausalNode(child)),
            (p.ConvertedReadout(0, channel=0), p.ConvertedReadout(0, channel=1)),
            p.ConvertedReadout(1, channel=0),
        )
    source = p.realize_gp_columns(p.GPColumnRequest(rows=16), seed=37, task_id=41)
    original = p.GPRootDAGRecipe(source, base, (p.GPRootBinding(0, (0, 1)),))
    calibrated = p.calibrate_gp_root_dag(
        original, calibration_rows=40, calibration_seed=43, calibration_task_id=47
    )
    task = p.generate_task(calibrated, seed=37, task_id=41)
    metadata = inspect_task(task)["target_calibration"]
    assert metadata["statistics"] == json.loads(json.dumps(asdict(calibrated.calibration)))
    assert metadata["population_kind"] == "independent GP population"
    raw = np.asarray(original.source.draw.values) @ np.array([2.0, -1.0])
    expected = (raw - calibrated.calibration.center[0]) / calibrated.calibration.scale[0]
    np.testing.assert_allclose(task.targets, expected, atol=1e-12)
    assert "target_calibration" not in inspect_task(p.generate_task(original, seed=37, task_id=41))
    assert "target_calibration" not in inspect_task(p.generate_task(base, seed=37, task_id=41))
