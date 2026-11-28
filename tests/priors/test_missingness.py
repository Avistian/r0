"""Independent probability/draw oracles for explicit observational masks."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.priors.lib.generator_snapshot import snapshot_generator


@pytest.mark.parametrize(
    "mode", ["mcar", "row", "block", "mar", "mnar-self", "mnar-target", "mnar-latent"]
)
@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_missingness_probability_and_rng_oracle(
    mode: str, engine: type[np.random.BitGenerator]
) -> None:
    x = np.arange(21, dtype=np.float64).reshape(7, 3) / 10 - 1
    logistic = mode.startswith("mn") or mode == "mar"
    plan = p.MissingnessPlan(
        mode,
        (0, 2),
        probability=None if logistic else 0.3,
        drivers=(1,) if mode == "mar" else (),
        coefficients=(2.0,) if mode == "mar" else (),
        intercept=-0.5 if logistic else 0.0,
        block_rows=3 if mode == "block" else 1,
    )
    driver = np.linspace(-2, 2, 7)
    target = driver if mode == "mnar-target" else None
    latent = driver if mode == "mnar-latent" else None
    result = p.draw_missingness(
        plan, x, rng=np.random.Generator(engine(30)), target=target, latent=latent
    )
    oracle = np.random.Generator(engine(30))
    if mode == "row":
        uniforms = np.broadcast_to(oracle.random((7, 1)), (7, 2))
    elif mode == "block":
        uniforms = np.repeat(oracle.random((3, 2)), 3, axis=0)[:7]
    else:
        uniforms = oracle.random((7, 2))
    if mode == "mar":
        logits = 2 * x[:, 1, None] - 0.5
    elif mode == "mnar-self":
        logits = x[:, (0, 2)] - 0.5
    else:
        logits = driver[:, None] - 0.5
    probability = (
        np.broadcast_to(1 / (1 + np.exp(-logits)), (7, 2)) if logistic else np.full((7, 2), 0.3)
    )
    np.testing.assert_allclose(result.probabilities[:, (0, 2)], probability, rtol=1e-14)
    np.testing.assert_array_equal(result.mask[:, (0, 2)], uniforms < probability)
    assert not result.mask[:, 1].any()
    assert not result.probabilities[:, 1].any()
    assert result.final == snapshot_generator(oracle)
    replay = p.draw_missingness(plan, x, rng=result.initial.restore(), target=target, latent=latent)
    assert replay.binding == result.binding
    np.testing.assert_array_equal(replay.mask, result.mask)
    altered = p.draw_missingness(
        plan, x + 1, rng=result.initial.restore(), target=target, latent=latent
    )
    assert altered.binding != result.binding


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_zero_and_full_missingness_are_retained(probability: float) -> None:
    result = p.draw_missingness(
        p.MissingnessPlan("mcar", (0, 1), probability=probability),
        np.ones((3, 2)),
        rng=np.random.default_rng(30),
    )
    assert bool(result.mask.all()) == bool(probability)
    assert bool(result.mask.any()) == bool(probability)


def test_missingness_rejection_preserves_rng() -> None:
    rng = np.random.default_rng(30)
    before = copy.deepcopy(rng.bit_generator.state)
    x = np.ones((3, 2))
    with pytest.raises(ValueError, match="workspace"):
        p.draw_missingness(
            p.MissingnessPlan("mcar", (0,), probability=0.5),
            x,
            rng=rng,
            limits=replace(p.CausalLimits(), working_bytes=1),
        )
    assert rng.bit_generator.state == before
    with pytest.raises(ValueError, match="overflow"):
        p.draw_missingness(p.MissingnessPlan("mnar-self", (0,), slope=1e308), x * 100, rng=rng)
    assert rng.bit_generator.state == before
    with pytest.raises(ValueError, match="unmasked"):
        p.MissingnessPlan("mar", (0,), drivers=(0,), coefficients=(1.0,))


def test_block_larger_than_table_uses_one_shared_block() -> None:
    result = p.draw_missingness(
        p.MissingnessPlan("block", (0, 1), probability=0.5, block_rows=20000),
        np.ones((2, 2)),
        rng=np.random.default_rng(30),
    )
    expected = np.random.default_rng(30).random((1, 2)) < 0.5
    np.testing.assert_array_equal(result.mask, np.repeat(expected, 2, axis=0))


@pytest.mark.parametrize("mode", ["column", "rectangle", "monotone"])
@pytest.mark.parametrize("probability", [0.0, 0.3, 1.0])
def test_structured_masks_follow_explicit_draw_oracle(mode: str, probability: float) -> None:
    # Non-sorted columns deliberately define monotone dropout order.
    plan = p.MissingnessPlan(
        mode, (2, 0), probability=probability, block_rows=3 if mode == "rectangle" else 1
    )
    x = np.arange(21, dtype=np.float64).reshape(7, 3)
    result = p.draw_missingness(plan, x, rng=np.random.default_rng(37))
    oracle = np.random.default_rng(37)
    marginal = np.full((7, 2), probability)
    if mode == "column":
        draws = np.repeat(oracle.random((1, 2)), 7, axis=0)
    elif mode == "rectangle":
        draws = np.repeat(np.repeat(oracle.random((3, 1)), 3, axis=0)[:7], 2, axis=1)
    else:
        draws = np.repeat(oracle.random((7, 1)), 2, axis=1)
        marginal[:, 1] = probability + (1 - probability) * probability
    np.testing.assert_array_equal(result.mask[:, (2, 0)], draws < marginal)
    np.testing.assert_allclose(result.probabilities[:, (2, 0)], marginal, atol=1e-15, rtol=0)
    assert not result.mask[:, 1].any()
    assert result.final == snapshot_generator(oracle)
    if mode == "monotone":
        assert not (result.mask[:, 2] & ~result.mask[:, 0]).any()
    replay = p.draw_missingness(plan, x, rng=result.initial.restore())
    np.testing.assert_array_equal(replay.mask, result.mask)


def test_monotone_small_hazard_preserves_nonzero_marginals() -> None:
    result = p.draw_missingness(
        p.MissingnessPlan("monotone", (0, 1), probability=1e-20),
        np.zeros((1, 2)),
        rng=np.random.default_rng(1),
    )
    np.testing.assert_allclose(result.probabilities, [[1e-20, 2e-20]], rtol=1e-15, atol=0)
    with pytest.raises(ValueError, match="block_rows"):
        p.MissingnessPlan("column", (0,), probability=0.5, block_rows=2)
