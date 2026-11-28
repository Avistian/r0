"""Source selector laws, per-layer randomness and composed numerical behavior."""

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

SIMPLE = (
    "tanh",
    "leaky_relu",
    "elu",
    "identity",
    "selu",
    "silu",
    "relu",
    "softplus",
    "relu6",
    "hardtanh",
    "sign_positive",
    "gaussian_bump",
    "exp",
    "sqrt_abs",
    "interval_abs_le_1",
    "sin",
    "square",
    "abs",
)
SAMPLE = np.array([[-2.0], [-1.0], [1.0], [2.0]])


def seed_for_slot(slot: int, total: int) -> int:
    return next(
        seed for seed in range(10000) if np.random.default_rng(seed).integers(total) == slot
    )


def test_every_simple_type_is_selectable_without_scaling() -> None:
    for index, kind in enumerate(SIMPLE):
        pipelines = p.sample_tabicl_activation_pipelines(
            (SAMPLE, SAMPLE),
            calibration_ids=("a", "b"),
            rng=np.random.default_rng(seed_for_slot(index, 18)),
            random_functions=False,
            scale=False,
            diverse=False,
        )
        assert tuple(x.stages for x in pipelines) == ((p.CausalActivation(kind),),) * 2


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_source_selection_law(engine: type[np.random.BitGenerator]) -> None:
    pool = SIMPLE + ("tabicl-code-fourier-v1",) * 10
    for seed in range(8):
        oracle = np.random.Generator(engine(seed))
        slot = int(oracle.integers(56))
        expected = (
            tuple(pool[int(oracle.integers(28))] for _ in range(3))
            if slot >= 28
            else (pool[slot],) * 3
        )
        pipelines = p.sample_tabicl_activation_pipelines(
            (SAMPLE,) * 3,
            calibration_ids=("a", "b", "c"),
            rng=np.random.Generator(engine(seed)),
            scale=False,
        )
        assert tuple(x.stages[-1].kind for x in pipelines) == expected
        for pipeline in pipelines:
            assert len(pipeline.stages) == (
                2 if pipeline.stages[-1].kind.endswith("fourier-v1") else 1
            )


@pytest.mark.parametrize("profile", ["paper", "pinned-code"])
@pytest.mark.parametrize("individual", [False, True])
def test_shared_fourier_has_fresh_parameters_and_correct_stage_order(
    profile: str, individual: bool
) -> None:
    sample = np.concatenate((SAMPLE, SAMPLE * 1e-5 + 3), axis=1)
    pipelines = p.sample_tabicl_activation_pipelines(
        (sample, sample),
        calibration_ids=("a", "b"),
        rng=np.random.default_rng(seed_for_slot(18, 56)),
        profile=profile,
        individual=individual,
    )
    assert pipelines[0].stages[-1] != pipelines[1].stages[-1]
    query = np.array([[0.5, 3.000002], [2.0, 2.99997]])
    for pipeline in pipelines:
        assert tuple(stage.kind for stage in pipeline.stages) == (
            "standardize",
            "tabicl-random-scale-v1",
            "standardize",
            "tabicl-paper-fourier-v1" if profile == "paper" else "tabicl-code-fourier-v1",
        )
        constants = dict(pipeline.stages[1].parameters)
        count = 2 if individual else 1
        a = np.array([constants[f"normal_scale_{i}"] for i in range(count)])
        b = np.array([constants[f"normal_shift_{i}"] for i in range(count)])
        scaled = np.exp(2 * a) * ((sample - sample.mean(0)) / (sample.std(0, ddof=1) + 1e-6) + b)
        x = np.exp(2 * a) * ((query - sample.mean(0)) / (sample.std(0, ddof=1) + 1e-6) + b)
        z = (x - scaled.mean(0)) / (scaled.std(0, ddof=1) + 1e-6)
        params = dict(pipeline.stages[-1].parameters)
        frequency = np.array([params[f"frequency_{i}"] for i in range(256)])
        phase = np.array([params[f"phase_{i}"] for i in range(256)])
        normal = np.array([params[f"normal_{i}"] for i in range(256)])
        weights = np.power(frequency, -math.exp(params["decay_draw"]))
        weights = weights / np.linalg.norm(weights) * normal
        expected = np.sum(np.sin(z[..., None] * frequency + phase) * weights, axis=-1)
        actual = torch.tensor(query)
        for stage in pipeline.stages:
            actual = stage.apply(actual)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-9, rtol=2e-9)


def test_selector_limits_preserve_rng() -> None:
    rng = np.random.default_rng(28)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="limit"):
        p.sample_tabicl_activation_pipelines(
            (SAMPLE,) * 2,
            calibration_ids=("a", "b"),
            rng=rng,
            limits=replace(p.CausalLimits(), parameters=800),
        )
    assert rng.bit_generator.state == before


def test_selected_pipeline_dataset_archive(tmp_path: Path) -> None:
    pipeline = p.sample_tabicl_activation_pipelines(
        (SAMPLE,),
        calibration_ids=("archive",),
        rng=np.random.default_rng(seed_for_slot(18, 56)),
    )[0]
    recipe = p.FunctionalCausalRecipe(
        p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        (
            p.FunctionalCausalNode(
                (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
            ),
            p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((1.0,),)), pipeline=pipeline),
        ),
        ((0, 0),),
        ((1, 0),),
    )
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    task = p.generate_task(restored, seed=28, task_id=2**54 + 1)
    path = tmp_path / "selected.zip"
    g.save_group(path, [(2**54 + 1, task)], max_bytes=4 * 1024**2)
    saved = next(g.load_group(path))
    assert saved.recipe_json == task.recipe_json
    assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] < 1e-12


def test_all_ten_fourier_slots_are_present() -> None:
    for slot in range(18, 28):
        pipeline = p.sample_tabicl_activation_pipelines(
            (SAMPLE,),
            calibration_ids=("weighted",),
            rng=np.random.default_rng(seed_for_slot(slot, 28)),
            diverse=False,
            scale=False,
        )[0]
        assert pipeline.stages[-1].kind == "tabicl-code-fourier-v1"


def test_late_failure_rolls_back_selection_and_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    from r0.modules.priors.lib import tabicl_fourier

    def fail(**kwargs: object) -> p.CausalActivation:
        raise ValueError("injected Fourier construction failure")

    monkeypatch.setattr(tabicl_fourier, "sample_tabicl_fourier_activation", fail)
    rng = np.random.default_rng(seed_for_slot(18, 56))
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="injected"):
        p.sample_tabicl_activation_pipelines((SAMPLE,), calibration_ids=("failure",), rng=rng)
    assert rng.bit_generator.state == before
