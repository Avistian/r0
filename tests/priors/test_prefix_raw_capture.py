"""Fresh ancestor samples retain every raw innovation, including rejected proposals."""

import copy

import numpy as np
import pytest
import torch

from r0.modules import priors


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
@pytest.mark.parametrize(
    ("family", "truncated"),
    [
        (family, truncated)
        for family in ("gaussian", "uniform", "student_t", "mixture")
        for truncated in (False, True)
    ]
    + [("ball", False)],
)
def test_prefix_capture_replays_original_draws_and_commits_one_calibration_stream(
    family: str, truncated: bool, engine: str
) -> None:
    inputs = priors.InputDistribution(
        distribution="gaussian" if family == "ball" else family,
        bound=0.25 if truncated else None,
    )
    exogenous = priors.ExogenousDistribution(
        inputs=inputs,
        geometry="ball" if family == "ball" else "independent",
        factor=((1.0, 2.0), (0.0, 1.0)),
    )
    root = priors.FunctionalCausalNode(
        (),
        priors.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
        root=priors.JointRootBase("ball", 2),
        exogenous=exogenous,
        noise_scale=(1.0, 2.0),
    )
    child = priors.FunctionalCausalNode((0,), priors.FrozenLinearFunction(((3.0, 4.0),)))
    prefix = priors.FunctionalPrefix((root, child), rows=5)
    generator = getattr(np.random, engine)
    rng = np.random.Generator(generator(29))
    control, replay = copy.deepcopy(rng), copy.deepcopy(rng)
    expected = priors.evaluate_functional_prefix(prefix, rng=lambda *keys: control)
    capture = priors.RawDrawLedger()
    actual = priors.sample_functional_prefix(prefix, rng=rng, owner=("fit",), capture=capture)
    for left, right in zip(actual.states, expected.states, strict=True):
        assert torch.equal(left, right)
    for left, right in zip(actual.readouts, expected.readouts, strict=True):
        assert torch.equal(left, right)
    records = capture.freeze()
    assert all(record.stream == "calibration" for record in records)
    assert all(record.owner[0] == "fit" for record in records)
    if truncated:
        assert any(record.shape == (256,) for record in records)
    for record in records:
        np.testing.assert_array_equal(record.replay(replay), record.array())
    next_values = control.random(8)
    np.testing.assert_array_equal(rng.random(8), next_values)
    np.testing.assert_array_equal(replay.random(8), next_values)

    # Every prior draw succeeds; rejection at the last record rolls back all of them.
    failed = np.random.Generator(generator(29))
    limited = priors.RawDrawLedger(limits=priors.RawDrawLimits(draws=len(records) - 1))
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.sample_functional_prefix(prefix, rng=failed, owner=("fit",), capture=limited)
    assert limited.freeze() == ()
    np.testing.assert_array_equal(failed.random(8), np.random.Generator(generator(29)).random(8))


def test_prefix_literal_mixture_equation_and_separate_successive_populations() -> None:
    root = priors.FunctionalCausalNode(
        (),
        priors.FrozenLinearFunction(((3.0,),)),
        root=priors.JointRootBase("gaussian", 1),
        exogenous=priors.ExogenousDistribution(priors.InputDistribution(distribution="mixture")),
        noise_scale=(2.0,),
    )
    prefix = priors.FunctionalPrefix((root,), rows=4)
    rng, oracle = np.random.default_rng(7), np.random.default_rng(7)
    capture = priors.RawDrawLedger()
    populations: list[torch.Tensor] = []
    for index in range(2):
        base = oracle.normal(size=(4, 1))
        normal = oracle.normal(size=4)
        component = oracle.integers(0, 2, size=4)
        expected = 3 * base[:, 0] + 2 * (normal + 2 * (2 * component - 1))
        result = priors.sample_functional_prefix(
            prefix,
            rng=rng,
            owner=(f"receiver:{index}",),
            capture=capture,
        )
        torch.testing.assert_close(result.states[0][:, 0], torch.tensor(expected))
        populations.append(result.states[0])
    assert not torch.equal(*populations)
    records = capture.freeze()
    assert [record.role for record in records] == [
        "root-base-values",
        "values",
        "mixture-components",
    ] * 2
    assert [record.owner[0] for record in records] == ["receiver:0"] * 3 + ["receiver:1"] * 3
    np.testing.assert_array_equal(rng.random(8), oracle.random(8))


def test_prefix_nonfinite_late_pipeline_rolls_back_draws_and_preserves_earlier_records() -> None:
    root = priors.FunctionalCausalNode(
        (),
        priors.FrozenTreeFunction(1, ((0,),), ((0.0,),), (((1e308,), (1e308,)),)),
        root=priors.JointRootBase("gaussian", 1),
        pipeline=priors.CausalNodePipeline(1, node_importance=2.0),
    )
    prefix = priors.FunctionalPrefix((root,), rows=4)
    rng, oracle = np.random.default_rng(7), np.random.default_rng(7)
    capture = priors.RawDrawLedger()
    with capture.transaction(np.random.default_rng(8)) as (prior, ledger):
        ledger.draw(prior, owner=("earlier",), role="values", distribution="normal", shape=(1,))
    earlier = capture.freeze()
    with pytest.raises(ValueError, match="nonfinite"):
        priors.sample_functional_prefix(prefix, rng=rng, owner=("failed",), capture=capture)
    assert capture.freeze() == earlier
    np.testing.assert_array_equal(rng.random(8), oracle.random(8))
