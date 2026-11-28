"""Pinned sample-standard-deviation convention with independent frozen calibration."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def test_tabicl_sample_standardization_and_constant_channel() -> None:
    calibration = np.array([[1.0, 7.0], [3.0, 7.0], [5.0, 7.0]])
    fitted = p.calibrate_standardization(
        calibration, calibration_id="independent-tabicl-fit", profile="tabicl"
    )
    assert fitted.calibration_version == "independent-tabicl-standardization-v1"
    assert fitted.mean == (3.0, 7.0)
    assert fitted.inverse_std == (1 / (2 + 1e-6), 1e6)
    query = np.array([[9.0, 7.5], [-3.0, 6.0]])
    expected = (query - calibration.mean(axis=0)) / (calibration.std(axis=0, ddof=1) + 1e-6)
    np.testing.assert_allclose(
        fitted.apply(torch.tensor(query)).numpy(), expected, rtol=1e-14, atol=1e-14
    )
    record = asdict(fitted)
    calibration[:] = -99
    fitted.apply(torch.tensor([[1e3, 7.0]], dtype=torch.float64))
    assert asdict(fitted) == record


def test_tabicl_one_row_rejects_but_existing_population_profile_remains_valid() -> None:
    with pytest.raises(ValueError, match="two|rows"):
        p.calibrate_standardization(np.array([[7.0]]), calibration_id="one", profile="tabicl")
    old = p.calibrate_standardization(np.array([[7.0]]), calibration_id="one")
    assert old.inverse_std == (10000.0,)
    assert old.calibration_version == "independent-standardization-v1"
    with pytest.raises(ValueError, match="profile"):
        p.calibrate_standardization(
            np.array([[7.0], [8.0]]), calibration_id="two", profile="unknown"
        )


def test_tabicl_extreme_representable_sample_std() -> None:
    fitted = p.calibrate_standardization(
        np.array([[0.0], [1.6e308]]), calibration_id="large", profile="tabicl"
    )
    actual = fitted.apply(torch.tensor([[-1.6e308]], dtype=torch.float64))
    torch.testing.assert_close(
        actual, torch.tensor([[-3 / math.sqrt(2)]], dtype=torch.float64), rtol=1e-14, atol=1e-14
    )


def test_tabicl_standardized_fourier_pipeline_archive(tmp_path: Path) -> None:
    independent = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    standard = p.calibrate_standardization(
        independent, calibration_id="independent-fourier-inputs", profile="tabicl"
    )
    activation = p.sample_tabicl_fourier_activation(rng=np.random.default_rng(28))
    scale = p.sample_tabicl_random_scale(1, rng=np.random.default_rng(29))
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode(
                (0,),
                p.FrozenLinearFunction(((1.0,),)),
                pipeline=p.CausalNodePipeline(1, (standard, scale, activation)),
            ),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    task = p.generate_task(restored, seed=28, task_id=2**54 + 1)
    normalized = (task.inputs.numpy() - independent.mean(axis=0)) / (
        independent.std(axis=0, ddof=1) + 1e-6
    )
    expected = activation.apply(scale.apply(torch.tensor(normalized)))[:, 0]
    torch.testing.assert_close(task.targets, expected, rtol=1e-12, atol=1e-12)
    path = tmp_path / "standardized-fourier.zip"
    g.save_group(path, [(2**54 + 1, task)], max_bytes=4 * 1024**2)
    saved = next(g.load_group(path))
    assert saved.recipe_json == task.recipe_json
    assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] < 1e-12
