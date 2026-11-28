"""Independent source equations for the two explicitly distinct Fourier laws."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from r0.modules import priors as p


@pytest.mark.parametrize("profile", ["paper", "pinned-code"])
@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_source_fourier_draws_and_equation(profile: str, engine: Any) -> None:
    rng = np.random.Generator(engine(28))
    oracle = np.random.Generator(engine(28))
    count = 256
    frequencies = oracle.uniform(0, count, count)
    phases = oracle.uniform(0, 2 * np.pi, count)
    bounds = (0.7, 3.0) if profile == "paper" else (np.log(0.7), np.log(3.0))
    decay = float(oracle.uniform(*bounds))
    gaussian = oracle.normal(size=count)
    factors: NDArray[np.float64] = frequencies ** (-np.exp(decay))
    coefficients: NDArray[np.float64] = gaussian * factors / float(np.linalg.norm(factors))
    activation = p.sample_tabicl_fourier_activation(rng=rng, profile=profile)
    x = np.array([[-3.0, 0.2], [0.0, 1.0], [2.0, 10.0]])
    expected: NDArray[np.float64] = (
        np.sin(x[..., None] * frequencies + phases) * coefficients
    ).sum(axis=-1)
    actual = activation.apply(torch.tensor(x, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert json.dumps(rng.bit_generator.state, default=lambda value: value.tolist()) == json.dumps(
        oracle.bit_generator.state, default=lambda value: value.tolist()
    )
    fields = asdict(activation)
    fields["parameters"] = tuple(tuple(pair) for pair in fields["parameters"])
    assert p.CausalActivation(**fields) == activation


def test_fourier_limits_and_zero_frequency_are_explicit_failures() -> None:
    rng = np.random.default_rng(9)
    before = json.dumps(rng.bit_generator.state)
    with pytest.raises(ValueError, match="limit"):
        p.sample_tabicl_fourier_activation(rng=rng, limits=p.CausalLimits(parameters=1))
    assert json.dumps(rng.bit_generator.state) == before
    sample = p.sample_tabicl_fourier_activation(rng=rng)
    fields = dict(sample.parameters)
    fields["frequency_0"] = 0.0
    with pytest.raises(ValueError, match="frequency"):
        p.CausalActivation(sample.kind, tuple(fields.items()))


@pytest.mark.parametrize("profile", ["paper", "pinned-code"])
def test_fourier_activation_network_and_causal_pipeline_replay(
    profile: str, tmp_path: Path
) -> None:
    from r0.modules import generation as g

    activation = p.sample_tabicl_fourier_activation(rng=np.random.default_rng(17), profile=profile)
    controls = p.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    network = p.FrozenNetworkFunction(
        1, (p.NetworkBlock((0,), (p.NetworkAffine(((1.0,),), (0.0,), activation=activation),)),)
    )
    recipes = (
        p.NetworkRecipe(controls, network),
        p.FunctionalCausalRecipe(
            controls,
            (
                p.FunctionalCausalNode(
                    (), p.FrozenLinearFunction(((1.0,),)), p.JointRootBase("gaussian", 1)
                ),
                p.FunctionalCausalNode(
                    (0,),
                    p.FrozenLinearFunction(((1.0,),)),
                    pipeline=p.CausalNodePipeline(1, (activation,)),
                ),
            ),
            ((0, 0),),
            ((1, 0),),
        ),
    )
    for i, recipe in enumerate(recipes):
        restored = p.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
        assert restored == recipe
        task = p.generate_task(restored, seed=28, task_id=2**54 + 3)
        torch.testing.assert_close(
            task.targets, activation.apply(task.inputs)[:, 0], rtol=1e-12, atol=1e-12
        )
        path = tmp_path / f"fourier-{i}.zip"
        g.save_group(path, [(2**54 + 3, task)], max_bytes=4 * 1024**2)
        saved = next(g.load_group(path))
        assert saved.recipe_json == task.recipe_json
        assert p.explain_generation(saved, [15, 0, 15])["signal_max_error"] < 1e-12
