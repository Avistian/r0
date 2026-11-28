"""Explicit projection/redundancy maps preserve targets and archive replay."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def stage() -> p.MeasurementStage:
    # On selected columns (2,0): output2=2*old2-old0+3; output0=old2.
    return p.MeasurementStage(
        "linear-mix",
        (2, 0),
        (("w_0_0", 2.0), ("w_0_1", -1.0), ("bias_0", 3.0), ("w_1_0", 1.0), ("w_1_1", 0.0)),
    )


def test_local_column_order_and_hand_projection() -> None:
    values = np.array([[1.0, 10.0, 4.0], [2.0, 20.0, 5.0]])
    result = p.draw_measurements((stage(),), values, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(result.values, [[4.0, 10.0, 10.0], [5.0, 20.0, 11.0]])
    np.testing.assert_array_equal(values, [[1.0, 10.0, 4.0], [2.0, 20.0, 5.0]])
    assert result.initial == result.final
    identity = p.draw_measurements(
        (p.MeasurementStage("linear-mix", (2, 0)),), values, rng=np.random.default_rng(3)
    )
    np.testing.assert_array_equal(identity.values, values)


def test_redundancy_noise_order_and_target_preservation(tmp_path: Path) -> None:
    task = p.generate_task(
        p.LinearRecipe(rows=16, context_rows=8, features=3, signal_features=3, active_features=3),
        seed=30,
        task_id=0,
    )
    stages = (
        p.MeasurementStage("linear-mix", (0, 1), (("w_1_0", 1.0), ("w_1_1", 0.0))),
        p.MeasurementStage("affine", (2,), (("scale", 0.0),)),
        p.MeasurementStage("gaussian-noise", (2,), (("scale", 0.2),)),
    )
    observed = p.observe_missingness(
        task, p.MissingnessPlan("mcar", (0,), probability=0), seed=42, measurements=stages
    )
    torch.testing.assert_close(observed.inputs[:, 0], observed.inputs[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(observed.targets, task.targets, rtol=0, atol=0)
    assert not torch.equal(observed.inputs[:, 2], task.inputs[:, 2])
    path = tmp_path / "mixed.zip"
    g.save_group(path, [(0, observed)], max_bytes=8000000)
    restored = p.restore_observation_source(next(g.load_group(path)))
    torch.testing.assert_close(restored.inputs, task.inputs, rtol=0, atol=0)


def test_admission_and_nonfinite_failure_preserve_rng() -> None:
    with pytest.raises(ValueError, match="unknown"):
        p.MeasurementStage("linear-mix", (0, 1), (("w_2_0", 1.0),))
    rng = np.random.default_rng(9)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="parameters"):
        p.draw_measurements(
            (stage(),), np.ones((4, 3)), rng=rng, limits=replace(p.CausalLimits(), parameters=5)
        )
    assert rng.bit_generator.state == before
    overflow = p.MeasurementStage("linear-mix", (0,), (("w_0_0", 1e308),))
    with pytest.raises(ValueError, match="nonfinite"):
        p.draw_measurements(
            (p.MeasurementStage("gaussian-noise", (0,)), overflow), np.full((4, 1), 100.0), rng=rng
        )
    assert rng.bit_generator.state == before
