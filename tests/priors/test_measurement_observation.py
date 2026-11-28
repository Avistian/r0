"""Measurement-then-mask composition retains raw truth and exact source replay."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def stages() -> tuple[p.MeasurementStage, ...]:
    return (
        p.MeasurementStage("affine", (0,), (("scale", 2.0), ("shift", 1.0))),
        p.MeasurementStage("gaussian-noise", (0,), (("scale", 0.5),)),
        p.MeasurementStage("clip", (0,), (("lower", -1.0), ("upper", 1.0))),
    )


def test_ordered_observation_numeric_oracle_and_archive(tmp_path: Path) -> None:
    source = p.generate_task(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=30,
        task_id=7,
    )
    result = p.observe_missingness(
        source, p.MissingnessPlan("mnar-self", (0,), slope=0.7), seed=42, measurements=stages()
    )
    expected = source.inputs.numpy().copy()
    measurement_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([0x4D454153, 42, 30, 7]))
    )
    expected[:, [0]] = np.clip(
        2 * expected[:, [0]] + 1 + 0.5 * measurement_rng.normal(size=(16, 1)), -1, 1
    )
    probabilities = 1 / (1 + np.exp(-0.7 * expected[:, [0]]))
    mask_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x4F425356, 42, 30, 7])))
    mask = mask_rng.random((16, 1)) < probabilities
    assert result.observation_truth is not None
    assert result.missing_mask is not None
    np.testing.assert_allclose(
        result.observation_truth.probabilities[:, [0]].numpy(), probabilities, rtol=1e-14
    )
    np.testing.assert_array_equal(result.missing_mask[:, [0]].numpy(), mask)
    np.testing.assert_allclose(
        result.inputs[:, 0].numpy()[~mask[:, 0]], expected[:, 0][~mask[:, 0]], rtol=1e-14
    )
    path = tmp_path / "measured.zip"
    g.save_group(path, [(7, result)], max_bytes=8000000)
    saved = next(g.load_group(path))
    assert p.restore_observation_source(saved).episode() == source.episode()
    altered = result.observation_truth.raw_inputs.clone()
    altered[0, 0] += 100
    with pytest.raises(ValueError, match="measurement receipt"):
        replace(result, observation_truth=replace(result.observation_truth, raw_inputs=altered))
    record = json.loads(result.recipe_json)
    assert record["observation"]["version"] == "r0-missing-observation-v2"
    changed = p.observe_missingness(
        source,
        p.MissingnessPlan("mnar-self", (0,), slope=0.7),
        seed=42,
        measurements=tuple(reversed(stages())),
    )
    assert changed.source_task_id != result.source_task_id


def test_measurements_do_not_silently_change_declared_category_codes() -> None:
    source = p.generate_task(p.LinearRecipe(rows=16, context_rows=8), seed=30, task_id=0)
    source = replace(
        source,
        inputs=torch.zeros_like(source.inputs),
        feature_schema=(p.TrainingFeature("count"),) * source.inputs.shape[1],
    )
    with pytest.raises(ValueError, match="continuous"):
        p.observe_missingness(
            source, p.MissingnessPlan("mcar", (0,), probability=0), seed=42, measurements=stages()
        )


def test_guarded_measurement_config_and_resume(tmp_path: Path) -> None:
    from dataclasses import asdict

    source = p.LinearRecipe(rows=16, context_rows=8)
    observation = g.ObservationRequest(
        p.MissingnessPlan("mcar", (0,), probability=0.3), seed=42, measurements=stages()
    )
    request = g.RunRequest(
        source, seed=30, count=2, max_seconds=30, max_bytes=8000000, observation=observation
    )
    assert g.request_from_dict(json.loads(json.dumps(asdict(request)))) == request
    run = g.run_guarded(request, tmp_path, workers=2)
    assert run.returncode == 0, run.stderr
    assert run.summary is not None
    assert run.summary["completed"] == 2
    resumed = g.run_guarded(request, tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
