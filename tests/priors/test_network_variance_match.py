"""Independent matching populations never borrow emitted query statistics."""

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors as p


def base() -> p.NetworkRecipe:
    return p.NetworkRecipe(
        p.LinearRecipe(rows=21, context_rows=9, features=2, signal_features=2, active_features=2),
        p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, -3.0),), (0.5,)),)),)
        ),
        p.JointRootBase("ball", 2),
    )


def test_independent_population_literal_matching_and_archive():
    original = base()
    match = p.fit_network_variance_match(
        original, before_inputs=(0, 1), after_inputs=(1,), rows=37, rng=np.random.default_rng(71)
    )
    x = np.asarray(match.population)
    before = 2 * x[:, 0] - 3 * x[:, 1] + 0.5
    after = -3 * x[:, 1] + 0.5
    np.testing.assert_allclose(match.before_scores, before[:, None], atol=1e-15)
    np.testing.assert_allclose(match.after_scores, after[:, None], atol=1e-15)
    np.testing.assert_allclose(match.before_mean, [before.mean()], atol=1e-15)
    np.testing.assert_allclose(match.after_std, [after.std()], atol=1e-15)
    view = p.MatchedNetworkInputView(
        original, replace(original.task, features=4, active_features=1), (1,), matching=match
    )
    emitted = p.generate_task(view, seed=19, task_id=23)
    expected = (-3 * emitted.inputs[:, 1].numpy() + 0.5) - after.mean()
    expected = expected / after.std() * before.std() + before.mean()
    np.testing.assert_allclose(emitted.scores[:, 0].numpy(), expected, atol=1e-14)
    assert p.recipe_from_dict(asdict(view)) == view
    assert torch.equal(p.evaluate_signal(emitted, emitted.inputs), emitted.scores)
    assert asdict(view.base) == asdict(original)
    assert "estimated" in emitted.mechanism_json


def test_constant_mask_failure_is_atomic():
    original = replace(
        base(),
        network=p.FrozenNetworkFunction(
            2, (p.NetworkBlock((0,), (p.NetworkAffine(((2.0, 0.0),), (0.5,)),)),)
        ),
    )
    rng = np.random.default_rng(3)
    state = str(rng.bit_generator.state)
    with pytest.raises(ValueError, match="constant|zero|dispersion"):
        p.fit_network_variance_match(
            original, before_inputs=(0, 1), after_inputs=(1,), rows=32, rng=rng
        )
    assert str(rng.bit_generator.state) == state


@pytest.mark.parametrize(
    "field",
    [
        "population",
        "before_scores",
        "after_scores",
        "before_mean",
        "after_std",
        "raw",
        "final",
        "digest",
    ],
)
def test_matching_archive_rejects_tampering(field: str):
    import json

    original = base()
    match = p.fit_network_variance_match(
        original, before_inputs=(0, 1), after_inputs=(1,), rows=13, rng=np.random.default_rng(7)
    )
    view = p.MatchedNetworkInputView(
        original, replace(original.task, active_features=1), (1,), matching=match
    )
    record = json.loads(json.dumps(asdict(view)))
    payload = record["matching"]
    if field in ("population", "before_scores", "after_scores"):
        payload[field][0][0] += 0.125
    elif field in ("before_mean", "after_std"):
        payload[field][0] += 0.125
    elif field == "raw":
        payload[field] = []
    elif field == "final":
        payload[field] = payload["initial"]
    else:
        payload[field] = "0" * 64
    with pytest.raises(ValueError, match="matching|limit|population|mask"):
        p.recipe_from_dict(record)


@pytest.mark.parametrize("family", ["gaussian", "uniform", "covariance"])
def test_shared_population_multiclass_numpy_oracle(family: str):
    original = p.NetworkRecipe(
        p.LinearRecipe(
            rows=29,
            context_rows=11,
            features=2,
            signal_features=2,
            active_features=2,
            task_type="multiclass",
            classes=3,
        ),
        p.FrozenNetworkFunction(
            2,
            (
                p.NetworkBlock(
                    (0,),
                    (p.NetworkAffine(((2.0, -3.0), (1.0, 2.0), (-4.0, 1.0)), (0.5, -1.0, 2.0)),),
                ),
            ),
        ),
        p.JointRootBase(
            family,
            2,
            factor=((1.0, 0.0), (0.5, 2.0)) if family == "covariance" else None,
            gaussian_factor=((2.0, 0.0), (1.0, 4.0)) if family == "covariance" else None,
            weights=p.UniformWeightVector((0.5, 0.5)) if family == "covariance" else None,
            covariance_base="gaussian" if family == "covariance" else None,
        ),
    )
    rng = np.random.default_rng(818)
    oracle = np.random.default_rng(818)
    x = oracle.random((41, 2)) * 2 - 1 if family == "uniform" else oracle.normal(size=(41, 2))
    if family == "covariance":
        x = x @ np.array([[1.0, 0.0], [0.5, 2.0]]).T
    match = p.fit_network_variance_match(
        original, before_inputs=(0, 1), after_inputs=(0,), rows=41, rng=rng
    )
    np.testing.assert_array_equal(match.population, x)
    assert str(rng.bit_generator.state) == str(oracle.bit_generator.state)
    weights = np.array([[2.0, -3.0], [1.0, 2.0], [-4.0, 1.0]])
    before = x @ weights.T + [0.5, -1.0, 2.0]
    masked = x.copy()
    masked[:, 1] = 0
    after = masked @ weights.T + [0.5, -1.0, 2.0]
    expected = (after - after.mean(0)) / after.std(0) * before.std(0) + before.mean(0)
    np.testing.assert_allclose(match.apply(torch.tensor(after)).numpy(), expected, atol=1e-14)
    view = p.MatchedNetworkInputView(
        original, replace(original.task, active_features=1), (0,), matching=match
    )
    a = p.generate_task(view, seed=7, task_id=2**54 + 3)
    b = p.generate_task(replace(view, task=replace(view.task, rows=43)), seed=19, task_id=4)
    assert a.scores.shape == (29, 3)
    assert b.scores.shape == (43, 3)
    assert view.matching == match
    assert p.recipe_from_dict(asdict(view)) == view


def test_preallocation_limits_masks_and_binding_preserve_rng():
    from unittest.mock import patch

    original = base()
    rng = np.random.default_rng(7)
    state = str(rng.bit_generator.state)
    for limited in (
        replace(original, limits=replace(original.limits, parameters=200)),
        replace(original, limits=replace(original.limits, working_bytes=1_000_000)),
    ):
        with (
            patch(
                "r0.modules.priors.lib.network_variance_match.draw_root_base",
                side_effect=AssertionError("allocated before admission"),
            ),
            pytest.raises(ValueError, match="limit"),
        ):
            p.fit_network_variance_match(limited, before_inputs=(0, 1), after_inputs=(1,), rng=rng)
        assert str(rng.bit_generator.state) == state
    for mask in ((), (1, 1), (2,), (1, 0)):
        with pytest.raises(ValueError, match="mask"):
            p.fit_network_variance_match(original, before_inputs=(0, 1), after_inputs=mask, rng=rng)
        assert str(rng.bit_generator.state) == state
    match = p.fit_network_variance_match(original, before_inputs=(0, 1), after_inputs=(1,), rng=rng)
    with pytest.raises(ValueError, match="differs"):
        p.MatchedNetworkInputView(
            original, replace(original.task, active_features=1), (0,), matching=match
        )
    with pytest.raises(ValueError, match="finite"):
        match.apply(torch.empty((0, 1), dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        match.apply(torch.full((1, 1), torch.inf, dtype=torch.float64))


def test_sampled_source_custody_and_no_refit_on_replay():
    sampled = p.sample_network(
        p.NetworkSamplingRequest(2, (3, 1), families=("dense",)),
        named_rng=np.random.default_rng(7),
        raw_rng=np.random.default_rng(8),
    )
    original = replace(base(), network=sampled)
    rng = np.random.default_rng(61)
    state = str(rng.bit_generator.state)
    with pytest.raises(ValueError, match="replay"):
        p.fit_network_variance_match(
            replace(original, network=replace(sampled, raw=())),
            before_inputs=(0, 1),
            after_inputs=(1,),
            rng=rng,
        )
    assert str(rng.bit_generator.state) == state
    match = p.fit_network_variance_match(
        original, before_inputs=(0, 1), after_inputs=(1,), rows=31, rng=rng
    )
    view = p.MatchedNetworkInputView(
        original, replace(original.task, active_features=1), (1,), matching=match
    )
    restored = p.recipe_from_dict(asdict(view))
    assert restored == view
    a = p.generate_task(view, seed=71, task_id=9)
    b = p.generate_task(restored, seed=71, task_id=9)
    assert a.recipe_json == b.recipe_json
    assert a.mechanism_json == b.mechanism_json
    assert torch.equal(a.scores, b.scores)
