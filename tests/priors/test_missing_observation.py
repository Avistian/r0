"""Observed tasks retain replayable privileged masks without exposing raw values."""

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


@pytest.mark.parametrize(
    "mode", ["mcar", "row", "block", "mar", "mnar-self", "mnar-target", "mnar-latent"]
)
def test_observation_archive_replay_and_predictor_boundary(mode: str, tmp_path: Path) -> None:
    source = p.generate_task(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=30,
        task_id=7,
    )
    plan = p.MissingnessPlan(
        mode,
        (0,),
        probability=0.5 if mode in ("mcar", "row", "block") else None,
        drivers=(1,) if mode == "mar" else (),
        coefficients=(1.0,) if mode == "mar" else (),
        block_rows=3 if mode == "block" else 1,
    )
    latent = torch.linspace(-2, 2, 16, dtype=torch.float64) if mode == "mnar-latent" else None
    task = p.observe_missingness(
        source,
        plan,
        seed=42,
        latent=latent,
        latent_provenance="authored row driver" if latent is not None else None,
    )
    assert task.source_task_id != source.source_task_id
    assert task.episode_id != source.episode_id
    assert task.observation_truth is not None
    assert task.missing_mask is not None
    torch.testing.assert_close(task.observation_truth.raw_inputs, source.inputs, rtol=0, atol=0)
    torch.testing.assert_close(task.targets, source.targets, rtol=0, atol=0)
    path = tmp_path / "observed.zip"
    g.save_group(path, [(7, task)], max_bytes=8000000)
    saved = next(g.load_group(path))
    p.validate_missing_observation(saved)
    assert saved.episode() == task.episode()
    restored = p.restore_observation_source(saved)
    assert restored.source_task_id == source.source_task_id
    assert restored.episode() == source.episode()
    assert p.explain_generation(restored, [0, 15])["signal_max_error"] < 1e-12
    with pytest.raises(ValueError, match="privileged truth"):
        replace(task, observation_truth=None)
    with zipfile.ZipFile(path) as archive:
        with np.load(io.BytesIO(archive.read("7/predictor.npz"))) as arrays:
            assert set(arrays.files) == {
                "context_x",
                "query_x",
                "context_y",
                "context_missing",
                "query_missing",
            }
        with np.load(io.BytesIO(archive.read("7/truth.npz"))) as arrays:
            assert "observation_raw_inputs" in arrays.files
            assert "observation_probabilities" in arrays.files
            assert ("observation_driver" in arrays.files) == (
                mode in ("mnar-target", "mnar-latent")
            )
    damaged = task.observation_truth.raw_inputs.clone()
    damaged[0, 0] += 1
    with pytest.raises(ValueError, match="replay"):
        replace(task, observation_truth=replace(task.observation_truth, raw_inputs=damaged))
    with pytest.raises(ValueError, match="unmasked"):
        p.observe_missingness(task, plan, seed=43)
    altered = json.loads(task.recipe_json)
    altered["observation"]["seed"] = 43
    with pytest.raises(ValueError, match="metadata"):
        replace(task, recipe_json=json.dumps(altered))
