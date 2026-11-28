"""Cross-column interactions use frozen pre-stage inputs and explicit coefficients."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def stage() -> p.MeasurementStage:
    # Local order (2,0): y2=old2+2*old2*old0; y0=old0-old2^2+3.
    return p.MeasurementStage(
        "quadratic-mix", (2, 0), (("q_0_0_1", 2.0), ("q_1_0_0", -1.0), ("bias_1", 3.0))
    )


def test_hand_interactions_are_simultaneous_and_ordered_between_stages() -> None:
    raw = np.array([[1.0, 10.0, 4.0], [2.0, 20.0, 5.0]])
    draw = p.draw_measurements((stage(),), raw, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(draw.values, [[-12.0, 10.0, 12.0], [-20.0, 20.0, 25.0]])
    assert draw.initial == draw.final
    affine = p.MeasurementStage("affine", (0,), (("shift", 1.0),))
    before = p.draw_measurements((affine, stage()), raw, rng=np.random.default_rng(3))
    after = p.draw_measurements((stage(), affine), raw, rng=np.random.default_rng(3))
    assert not np.array_equal(before.values, after.values)
    np.testing.assert_array_equal(raw, [[1.0, 10.0, 4.0], [2.0, 20.0, 5.0]])


def test_interaction_archive_replay_keeps_source_and_targets(tmp_path: Path) -> None:
    task = p.generate_task(
        p.LinearRecipe(rows=16, context_rows=8, features=3, signal_features=3, active_features=3),
        seed=30,
        task_id=0,
    )
    observed = p.observe_missingness(
        task, p.MissingnessPlan("mcar", (0,), probability=0), seed=42, measurements=(stage(),)
    )
    torch.testing.assert_close(observed.targets, task.targets, rtol=0, atol=0)
    expected = task.inputs.numpy().copy()
    expected[:, 2] = (
        task.inputs[:, 2].numpy() + 2 * task.inputs[:, 2].numpy() * task.inputs[:, 0].numpy()
    )
    expected[:, 0] = task.inputs[:, 0].numpy() - task.inputs[:, 2].numpy() ** 2 + 3
    np.testing.assert_allclose(observed.inputs.numpy(), expected, rtol=1e-14, atol=1e-14)
    archive = tmp_path / "quadratic.zip"
    g.save_group(archive, [(0, observed)], max_bytes=8000000)
    restored = p.restore_observation_source(next(g.load_group(archive)))
    torch.testing.assert_close(restored.inputs, task.inputs, rtol=0, atol=0)


def test_bad_indices_aliases_and_resource_admission() -> None:
    for name in ("q_0_1_0", "q_0_0_2", "q_00_0_1", "q_0_0"):
        with pytest.raises(ValueError, match="unknown"):
            p.MeasurementStage("quadratic-mix", (0, 1), ((name, 1.0),))
    with pytest.raises(ValueError, match="unknown"):
        p.MeasurementStage("linear-mix", (0, 1), (("q_0_0_1", 1.0),))
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="parameters"):
        p.draw_measurements(
            (stage(),), np.ones((4, 3)), rng=rng, limits=replace(p.CausalLimits(), parameters=7)
        )
    assert rng.bit_generator.state == state
    with pytest.raises(ValueError, match="nonfinite"):
        p.draw_measurements((stage(),), np.full((4, 3), 1e300), rng=rng)
    assert rng.bit_generator.state == state
