"""Independent sampling-law checks and frozen extension recipe replay."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def task() -> p.LinearRecipe:
    return p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)


@pytest.mark.parametrize("family", ["hard", "soft"])
@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_construction_draw_laws_and_replay(
    family: str, engine: type[np.random.BitGenerator]
) -> None:
    request = p.TreeExtensionRequest(task(), families=(family,), min_depth=1, max_depth=1)
    rng = np.random.Generator(engine(29))
    sample = p.sample_tree_extension(request, rng=rng)
    sample.validate_replay()
    assert p.sampled_tree_extension_from_dict(json.loads(json.dumps(asdict(sample)))) == sample
    assert sample.depth == 1
    assert sample.family == family
    oracle = np.random.Generator(engine(29))
    oracle.integers(1)
    oracle.integers(1, 2)
    weights = oracle.normal(size=(1, 2))
    weights *= oracle.random((1, 2)) < 0.5
    thresholds = oracle.uniform(-3, 3, 1)
    temperature = np.exp(oracle.uniform(np.log(0.05), np.log(2.0), 1))
    if family == "hard":
        assert isinstance(sample.recipe, p.FunctionalCausalRecipe)
        projection = sample.recipe.nodes[1].function
        assert isinstance(projection, p.FrozenLinearFunction)
        np.testing.assert_array_equal(projection.matrix, weights)
        function = sample.recipe.nodes[2].function
        assert isinstance(function, p.FrozenRecursiveForestFunction)
        assert function.forest.trees[0].nodes[0].threshold == thresholds[0]
    else:
        assert isinstance(sample.recipe, p.NetworkRecipe)
        gate = sample.recipe.function.blocks[-1].gate
        assert gate is not None
        np.testing.assert_allclose(gate.matrix[1][:2], weights[0] / temperature[0], rtol=1e-14)
        assert gate.bias[1] == pytest.approx(-thresholds[0] / temperature[0])
    restored = p.recipe_from_dict(json.loads(json.dumps(asdict(sample.recipe))))
    original = p.generate_task(sample.recipe, seed=29, task_id=9)
    replay = p.generate_task(restored, seed=29, task_id=9)
    torch.testing.assert_close(original.targets, replay.targets, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["hard", "soft"])
@pytest.mark.parametrize("depth", [0, 3])
def test_stump_and_deeper_portfolio_archive(family: str, depth: int, tmp_path: Path) -> None:
    sample = p.sample_tree_extension(
        p.TreeExtensionRequest(task(), families=(family,), min_depth=depth, max_depth=depth),
        rng=np.random.default_rng(29),
    )
    sample.validate_replay()
    generated = p.generate_task(sample.recipe, seed=29, task_id=2**54 + 1)
    path = tmp_path / "sampled.zip"
    g.save_group(path, [(2**54 + 1, generated)], max_bytes=16 * 1024**2)
    assert next(g.load_group(path)).recipe_json == generated.recipe_json
    with pytest.raises(ValueError, match="replay"):
        replace(sample, depth=depth + 1).validate_replay()


def test_request_limit_rejection_preserves_rng() -> None:
    rng = np.random.default_rng(29)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="limit"):
        p.sample_tree_extension(
            p.TreeExtensionRequest(task()), rng=rng, limits=replace(p.CausalLimits(), parameters=1)
        )
    assert rng.bit_generator.state == before


def test_late_compile_failure_rolls_back_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    from r0.modules.priors.lib import tree_extension_sampling

    def fail(*args: object, **kwargs: object) -> object:
        raise ValueError("injected compilation failure")

    monkeypatch.setattr(tree_extension_sampling, "compile_soft_tree", fail)
    rng = np.random.default_rng(29)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="injected"):
        p.sample_tree_extension(p.TreeExtensionRequest(task(), families=("soft",)), rng=rng)
    assert rng.bit_generator.state == before


def test_receipt_rejects_boolean_depth() -> None:
    sample = p.sample_tree_extension(
        p.TreeExtensionRequest(task(), min_depth=1, max_depth=1), rng=np.random.default_rng(29)
    )
    data = json.loads(json.dumps(asdict(sample)))
    data["depth"] = True
    with pytest.raises(ValueError, match="metadata"):
        p.sampled_tree_extension_from_dict(data)
