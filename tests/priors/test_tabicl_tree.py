"""Noncausal source geometry with explicit independent-fit adaptation."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def streams(seed: int = 28) -> dict[str, np.random.Generator]:
    return {
        name: np.random.default_rng(seed + i)
        for i, name in enumerate(
            (
                "architecture_rng",
                "parameter_rng",
                "input_rng",
                "target_rng",
                "backend_rng",
            )
        )
    }


@pytest.mark.parametrize("seed", [1, 28])
@pytest.mark.parametrize("presample", [False, True])
def test_tree_geometry_noise_and_archive(presample: bool, seed: int, tmp_path: Path) -> None:
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    rngs = streams(seed)
    graph = p.sample_tabicl_tree_graph(
        task,
        architecture_rng=rngs["architecture_rng"],
        parameter_rng=rngs["parameter_rng"],
        input_rng=rngs["input_rng"],
        target_rng=rngs["target_rng"],
        backend_rng=rngs["backend_rng"],
        fit_rows=32,
        pre_sample_noise_std=presample,
    )
    oracle = np.random.default_rng(seed)
    depth, width = int(oracle.integers(1, 3)), int(oracle.integers(3, 10))
    assert len(graph.nodes) == depth + 2
    assert graph.observed == ((0, 0), (0, 1))
    assert graph.target == ((depth + 1, 0),)
    assert graph.nodes[1].noise_scale == (0.0,) * width
    for index in range(2, depth + 2):
        channels = width if index <= depth else 1
        expected = (
            tuple(float(v) for v in np.abs(oracle.normal(0, 0.01, channels)))
            if presample
            else (0.01,) * channels
        )
        assert graph.nodes[index].noise_scale == expected
    assert rngs["architecture_rng"].random() == oracle.random()
    assert len(graph.fits) == depth + 1
    for fit in graph.fits:
        assert fit.realization.plan.rows == 32
        assert 2 <= fit.realization.plan.depth <= 4
        assert 1 <= fit.realization.plan.estimators <= 4
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(graph))))
    generated = p.generate_task(restored, seed=28, task_id=2**54 + 1)
    path = tmp_path / "tree.zip"
    g.save_group(path, [(2**54 + 1, generated)], max_bytes=8 * 1024**2)
    saved = next(g.load_group(path))
    assert saved.recipe_json == generated.recipe_json
    replay = p.generate_task(restored, seed=28, task_id=2**54 + 1)
    torch.testing.assert_close(replay.inputs, generated.inputs, rtol=0, atol=0)
    torch.testing.assert_close(replay.targets, generated.targets, rtol=0, atol=0)
    assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] < 1e-12


def test_tree_admission_preserves_all_streams() -> None:
    rngs = streams()
    before = {key: copy.deepcopy(value.bit_generator.state) for key, value in rngs.items()}
    with pytest.raises(ValueError, match="limit|channel"):
        p.sample_tabicl_tree_graph(
            p.LinearRecipe(),
            architecture_rng=rngs["architecture_rng"],
            parameter_rng=rngs["parameter_rng"],
            input_rng=rngs["input_rng"],
            target_rng=rngs["target_rng"],
            backend_rng=rngs["backend_rng"],
            limits=replace(p.CausalLimits(), node_channels=1),
        )
    assert {key: value.bit_generator.state for key, value in rngs.items()} == before


def test_backend_failure_rolls_back_architecture_and_fitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from r0.modules.priors.lib import sampled_xgboost_graph

    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("injected backend failure")

    monkeypatch.setattr(sampled_xgboost_graph, "fit_xgboost", fail)
    rngs = streams()
    before = {key: copy.deepcopy(value.bit_generator.state) for key, value in rngs.items()}
    with pytest.raises(ValueError, match="injected"):
        p.sample_tabicl_tree_graph(
            p.LinearRecipe(),
            architecture_rng=rngs["architecture_rng"],
            parameter_rng=rngs["parameter_rng"],
            input_rng=rngs["input_rng"],
            target_rng=rngs["target_rng"],
            backend_rng=rngs["backend_rng"],
            fit_rows=32,
        )
    assert {key: value.bit_generator.state for key, value in rngs.items()} == before
