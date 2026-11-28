"""Shared-input ensemble arithmetic and archived compiled composition."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def members() -> tuple[p.FunctionalCausalRecipe | p.NetworkRecipe, ...]:
    task = p.LinearRecipe(
        rows=32, context_rows=16, features=2, signal_features=2, active_features=2
    )
    return tuple(
        p.sample_tree_extension(
            p.TreeExtensionRequest(task, families=(family,), min_depth=1, max_depth=1),
            rng=np.random.default_rng(seed),
        ).recipe
        for family, seed in (("hard", 3), ("soft", 5))
    )


@pytest.mark.parametrize(
    ("weights", "reduction", "coefficients"),
    [
        ((), "mean", (0.5, 0.5)),
        ((1.0, 3.0), "mean", (0.25, 0.75)),
        ((2.0, -1.0), "sum", (2.0, -1.0)),
        ((0.0, 0.0), "sum", (0.0, 0.0)),
    ],
)
def test_mixed_tree_ensemble_oracle_and_archive(
    weights: tuple[float, ...], reduction: str, coefficients: tuple[float, ...], tmp_path: Path
) -> None:
    source = members()
    recipe = p.score_ensemble_recipe(source, weights=weights, reduction=reduction)
    assert p.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
    individual = [p.generate_task(m, seed=29, task_id=8) for m in source]
    generated = p.generate_task(recipe, seed=29, task_id=8)
    for member in individual:
        torch.testing.assert_close(member.inputs, generated.inputs, rtol=0, atol=0)
    expected = sum(c * m.targets for c, m in zip(coefficients, individual, strict=True))
    torch.testing.assert_close(generated.targets, expected, rtol=1e-13, atol=1e-13)
    path = tmp_path / "ensemble.zip"
    g.save_group(path, [(8, generated)], max_bytes=8 * 1024**2)
    saved = next(g.load_group(path))
    torch.testing.assert_close(saved.targets, generated.targets, rtol=0, atol=0)
    assert p.explain_generation(saved, [0, 31])["signal_max_error"] < 1e-12


def test_ensemble_rejects_mismatched_geometry_noise_and_invalid_weights() -> None:
    source = members()
    with pytest.raises(ValueError, match="weight sum"):
        p.score_ensemble_recipe(source, weights=(1.0, -1.0))
    with pytest.raises(ValueError, match="geometry"):
        p.score_ensemble_recipe(source, limits=replace(p.CausalLimits(), nodes=2))
    first = source[0]
    assert isinstance(first, p.FunctionalCausalRecipe)
    changed = replace(first, task=replace(first.task, noise_ratio=0.2))
    with pytest.raises(ValueError, match="identical"):
        p.score_ensemble_recipe((first, changed))
    noisy = replace(
        first, nodes=(replace(first.nodes[0], noise_scale=(0.1, 0.1)),) + first.nodes[1:]
    )
    with pytest.raises(ValueError, match="noise-free"):
        p.score_ensemble_recipe((noisy, noisy))


def test_vector_score_reduction_precedes_class_adapter() -> None:
    task = p.LinearRecipe(
        rows=32,
        context_rows=16,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=3,
    )
    matrix = ((1.0, 0.0), (0.0, 1.0), (-1.0, -1.0))
    source = tuple(
        p.NetworkRecipe(
            task,
            p.FrozenNetworkFunction(2, (p.NetworkBlock((0,), (p.NetworkAffine(matrix, bias),)),)),
        )
        for bias in ((1.0, 0.0, -1.0), (-2.0, 1.0, 3.0))
    )
    recipe = p.score_ensemble_recipe(source, weights=(1.0, 3.0))
    generated = p.generate_task(recipe, seed=29, task_id=8)
    expected = generated.inputs @ torch.tensor(matrix, dtype=torch.float64).T
    expected += torch.tensor((-1.25, 0.75, 2.0), dtype=torch.float64)
    torch.testing.assert_close(generated.scores, expected, rtol=1e-13, atol=1e-13)
