"""Independent source-law and frozen lifecycle checks for random-frequency sine."""

import copy
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.priors.lib.calibrated_activations import activation_from_dict


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_sine_draw_law_frozen_statistics_and_json(engine: type[np.random.BitGenerator]) -> None:
    rng, oracle = np.random.Generator(engine(28)), np.random.Generator(engine(28))
    calibration = np.array([[-3.0, 7.0], [-1.0, 7.0], [1.0, 7.0], [3.0, 7.0]])
    stages = p.sample_tabicl_sine(calibration, calibration_id="independent-sine", rng=rng)
    frequency = math.exp(oracle.uniform(math.log(0.1), math.log(100)))
    phase = oracle.uniform(0, 2 * math.pi)
    assert dict(stages[1].parameters) == {"input_scale": frequency, "input_shift": phase}
    assert rng.random() == oracle.random()
    query = np.array([[10.0, 7.000001], [-2.0, 7.0], [10.0, 7.000001]])
    expected = np.sin(
        frequency * (query - calibration.mean(0)) / (calibration.std(0, ddof=1) + 1e-6) + phase
    )
    restored = tuple(activation_from_dict(json.loads(json.dumps(asdict(s)))) for s in stages)
    calibration[:] = 99
    actual = torch.tensor(query)
    for stage in restored:
        actual = stage.apply(actual)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-12, rtol=2e-12)
    assert restored == stages


def test_invalid_sine_construction_preserves_stream() -> None:
    rng = np.random.default_rng(28)
    before = copy.deepcopy(rng.bit_generator.state)
    calibration = np.array([[-1.0], [1.0]])
    for bounds in ((0, 1), (2, 1), (1, float("inf"))):
        with pytest.raises(ValueError, match="bounds"):
            p.sample_tabicl_sine(
                calibration,
                calibration_id="bad",
                rng=rng,
                min_frequency=bounds[0],
                max_frequency=bounds[1],
            )
    with pytest.raises(ValueError, match="limit"):
        p.sample_tabicl_sine(
            calibration,
            calibration_id="small",
            rng=rng,
            limits=replace(p.CausalLimits(), parameters=3),
        )
    assert rng.bit_generator.state == before


def test_sine_generated_archive_and_trace(tmp_path: Path) -> None:
    calibration = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    stages = p.sample_tabicl_sine(
        calibration, calibration_id="archive-sine", rng=np.random.default_rng(14)
    )
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode(
                (0,), p.FrozenLinearFunction(((1.0,),)), pipeline=p.CausalNodePipeline(1, stages)
            ),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    task = p.generate_task(recipe, seed=28, task_id=2**54 + 1)
    params = dict(stages[1].parameters)
    x = task.inputs.numpy()
    expected = np.sin(
        params["input_scale"] * (x - calibration.mean(0)) / (calibration.std(0, ddof=1) + 1e-6)
        + params["input_shift"]
    )
    np.testing.assert_allclose(task.targets.numpy(), expected[:, 0], atol=2e-12, rtol=2e-12)
    path = tmp_path / "sine.zip"
    g.save_group(path, [(2**54 + 1, task)], max_bytes=4 * 1024**2)
    saved = next(g.load_group(path))
    assert saved.recipe_json == task.recipe_json
    assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] < 1e-12
