"""r0 observation extensions select propagated general-DAG states, never paper aliases."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p

PROFILE = "r0-propagated-block-v1"


def source() -> p.NetworkRecipe:
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    first = p.NetworkBlock(
        (0,), (p.NetworkAffine(((1.0, 2.0), (2.0, -1.0), (-1.0, 0.5)), (0.1, 0.2, -0.1)),)
    )
    second = p.NetworkBlock(
        (1, 0),
        (
            p.NetworkAffine(((1.0, 0.0, 0.5, 2.0, -1.0), (0.0, 1.0, -1.0, 0.5, 2.0)), (0.2, -0.3)),
            p.NetworkAffine(((0.0, 1.0, 0.0, -1.0, 1.0), (1.0, -1.0, 2.0, 0.0, 0.5)), (-0.1, 0.4)),
        ),
        gate=p.NetworkAffine(((0.1, -0.2, 0.3, 0.1, 0.0), (-0.1, 0.2, 0.0, 0.0, 0.2)), (0.0, 0.1)),
        skips=(p.NetworkProjection(0, p.NetworkAffine(((0.5, 0.0), (0.0, -0.5)), (0.0, 0.0))),),
    )
    final = p.NetworkBlock((2, 1), (p.NetworkAffine(((1.0, -2.0, 0.5, 1.0, -0.5),), (0.3,)),))
    return p.NetworkRecipe(task, p.FrozenNetworkFunction(2, (first, second, final)))


def root_rows() -> np.ndarray:
    key = json.dumps(
        ("functional-causal-v1", 19, 23, ("root-base", 0)), sort_keys=True, separators=(",", ":")
    )
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16)).normal(
        size=(16, 2)
    )


def literal(root: np.ndarray, features: tuple[int, ...] = (), clamps: np.ndarray | None = None):
    a = root @ np.array([[1, 2], [2, -1], [-1, 0.5]]).T + [0.1, 0.2, -0.1]

    def clamp(values: np.ndarray, offset: int) -> None:
        if clamps is not None:
            for col, index in enumerate(features):
                if offset <= index < offset + values.shape[1]:
                    values[:, index - offset] = clamps[:, col]

    clamp(a, 0)
    z = np.concatenate([a, root], axis=1)
    left = z @ np.array([[1, 0, 0.5, 2, -1], [0, 1, -1, 0.5, 2]]).T + [0.2, -0.3]
    right = z @ np.array([[0, 1, 0, -1, 1], [1, -1, 2, 0, 0.5]]).T + [-0.1, 0.4]
    logits = z @ np.array([[0.1, -0.2, 0.3, 0.1, 0], [-0.1, 0.2, 0, 0, 0.2]]).T + [0, 0.1]
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    b = weights[:, :1] * left + weights[:, 1:] * right + root * [0.5, -0.5]
    clamp(b, 3)
    c = np.concatenate([b, a], axis=1) @ np.array([[1, -2, 0.5, 1, -0.5]]).T + 0.3
    clamp(c, 5)
    return np.concatenate([a, b, c], axis=1)


@pytest.mark.parametrize("selection", ["block", "unrestricted"])
@pytest.mark.parametrize("target", ["internal", "final"])
@pytest.mark.parametrize("order", ["preserved", "permuted"])
def test_general_dag_selection_and_literal_state_values(
    selection: str, target: str, order: str
) -> None:
    src = source()
    request = p.NetworkObservationRequest(
        src, 2, profile=PROFILE, selection=selection, target=target, feature_order=order
    )
    result = p.sample_network_observations(request, rng=np.random.default_rng(7))
    oracle = np.random.default_rng(7)
    if selection == "block":
        start = [0, 1, 3][int(oracle.integers(0, 3, size=1)[0])]
        features = (start, start + 1)
        available = [i for i in range(6) if i not in features]
        targets = (5,) if target == "final" else (available[int(oracle.permutation(4)[0])],)
        if order == "permuted":
            features = tuple(features[int(i)] for i in oracle.permutation(2))
    else:
        perm = oracle.permutation(5 if target == "final" else 6)
        targets = (5,) if target == "final" else (int(perm[0]),)
        features = tuple(map(int, perm[:2] if target == "final" else perm[1:3]))
    if order == "preserved":
        features = tuple(sorted(features))
    assert (result.features, result.targets) == (features, targets)
    np.testing.assert_array_equal(result.final.restore().random(5), oracle.random(5))
    assert [(c.block, c.node, c.channel) for c in result.catalog] == [
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (1, 2, 0),
        (1, 2, 1),
        (2, 3, 0),
    ]
    recipe = p.NetworkObservationRecipe(src.task, result)
    task = p.generate_task(recipe, seed=19, task_id=23)
    expected = literal(root_rows())
    np.testing.assert_allclose(task.inputs.numpy(), expected[:, features], atol=1e-12)
    np.testing.assert_allclose(task.scores.numpy(), expected[:, targets], atol=1e-12)
    query = np.array([[-0.5, 0.75], [1.5, -2.0], [0.25, 0.5]])
    roots = np.repeat(root_rows()[4:5], 3, axis=0)
    reference = p.evaluate_observation_reference(
        recipe, torch.tensor(query), seed=19, task_id=23, row=4
    )
    np.testing.assert_allclose(
        reference.numpy(), literal(roots, features, query)[:, targets], atol=1e-12
    )
    assert p.recipe_from_dict(asdict(recipe)) == recipe
    trace = p.explain_generation(task, [15, 0, 15])
    assert trace["signal_max_error"] < 1e-12
    for column, (node, channel) in enumerate(recipe.runtime.observed):
        readout = next(n for n in trace["nodes"] if n["id"] == f"readout-{node}")
        np.testing.assert_allclose(
            np.array(readout["values"])[:, channel], task.inputs[[15, 0, 15], column].numpy()
        )


@pytest.mark.parametrize("profile", ["p1-paper-block-r0-v1", "p1-778f0d0-v1"])
def test_paper_profiles_still_reject_general_architectures(profile: str) -> None:
    with pytest.raises(ValueError, match="sequential"):
        p.NetworkObservationRequest(source(), 2, profile=profile)


def test_general_dag_archive_tamper_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = source()
    selected = p.sample_network_observations(
        p.NetworkObservationRequest(src, 2, profile=PROFILE), rng=np.random.default_rng(4)
    )
    recipe = p.NetworkObservationRecipe(src.task, selected)
    task = p.generate_task(recipe, seed=19, task_id=23)
    archive = tmp_path / "general.zip"
    g.save_group(archive, [(23, task)], max_bytes=256 * 1024**2)
    reopened = next(g.load_group(archive))
    assert reopened.recipe_json == task.recipe_json
    assert p.explain_generation(reopened, [15, 0])["signal_max_error"] < 1e-12
    changed = list(selected.catalog)
    changed[3] = replace(changed[3], node=1)
    with pytest.raises(ValueError, match="catalog"):
        replace(selected, catalog=tuple(changed))
    with pytest.raises(ValueError, match="raw draws"):
        replace(selected, features=tuple(reversed(selected.features)))

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("replay before admission")

    monkeypatch.setattr(type(selected.initial), "restore", forbidden)
    with pytest.raises(ValueError, match="limit|budget|cap"):
        p.recipe_from_dict(asdict(recipe), limits=replace(src.limits, working_bytes=1))
    with pytest.raises(ValueError, match="raw"):
        p.NetworkObservationRequest(src, 2, profile=PROFILE, raw_limits=p.RawDrawLimits(numbers=1))


@pytest.mark.parametrize(
    "family",
    ["dense", "individual-sparse", "block-sparse", "skip", "residual", "irregular", "gated"],
)
@pytest.mark.parametrize("noise", [False, True])
def test_every_architecture_with_variable_widths_and_postblock_noise(
    family: str, noise: bool
) -> None:
    task = source().task
    sampled = p.sample_network(
        p.NetworkSamplingRequest(2, (3, 2, 1), families=(family,)),
        named_rng=np.random.default_rng(71),
        raw_rng=np.random.default_rng(72),
    )
    network = p.NetworkRecipe(task, sampled)
    original: p.NetworkRecipe | p.NetworkInnovationRecipe = network
    if noise:
        original = p.with_network_innovations(
            network,
            p.NetworkInnovationPlan(
                tuple(
                    p.NetworkInnovationStage((0.25,) * width, (0.2,) * width) for width in (3, 2, 1)
                )
            ),
        )
    selected = p.sample_network_observations(
        p.NetworkObservationRequest(original, 2, profile=PROFILE), rng=np.random.default_rng(73)
    )
    recipe = p.NetworkObservationRecipe(task, selected)
    generated = p.generate_task(recipe, seed=19, task_id=23)
    replayed = p.generate_task(p.recipe_from_dict(asdict(recipe)), seed=19, task_id=23)
    assert torch.equal(generated.inputs, replayed.inputs)
    assert torch.equal(generated.scores, replayed.scores)
    p.explain_generation(generated, [15, 0, 15])
    ref = p.evaluate_observation_reference(
        recipe, generated.inputs[3:4], seed=19, task_id=23, row=3
    )
    torch.testing.assert_close(ref, generated.scores[3:4], atol=1e-12, rtol=1e-12)
    assert tuple(c.node for c in selected.catalog) == (
        (2, 2, 2, 4, 4, 6) if noise else (1, 1, 1, 2, 2, 3)
    )


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_selection_stream_engines_and_rejection_rollback(
    engine: type[np.random.BitGenerator], monkeypatch: pytest.MonkeyPatch
) -> None:
    request = p.NetworkObservationRequest(source(), 2, profile=PROFILE)
    rng = np.random.Generator(engine(91))
    result = p.sample_network_observations(request, rng=rng)
    assert p.network_observations_from_dict(asdict(result)) == result
    np.testing.assert_array_equal(rng.random(5), result.final.restore().random(5))
    state = repr(rng.bit_generator.state)

    def reject(*args: object, **kwargs: object):
        raise ValueError("injected receipt failure")

    monkeypatch.setattr(p.SampledNetworkObservations, "validate_replay", reject)
    with pytest.raises(ValueError, match="injected receipt"):
        p.sample_network_observations(request, rng=rng)
    assert repr(rng.bit_generator.state) == state


def test_final_width_and_block_capacity_are_not_silently_repaired() -> None:
    with pytest.raises(ValueError, match="final targets"):
        p.NetworkObservationRequest(source(), 2, outputs=2, target="final", profile=PROFILE)
    with pytest.raises(ValueError, match="within-layer"):
        p.NetworkObservationRequest(source(), 4, profile=PROFILE)
    for profile in ("p1-paper-block-r0-v1", "p1-778f0d0-v1"):
        base = source()
        sequential = p.FrozenNetworkFunction(
            2,
            (
                base.function.blocks[0],
                p.NetworkBlock((1,), (p.NetworkAffine(((1.0, 2.0, 3.0),), (0.0,)),)),
            ),
        )
        with pytest.raises(ValueError, match="variable-width"):
            p.NetworkObservationRequest(replace(base, network=sequential), 2, profile=profile)


@pytest.mark.parametrize("location", ["state", "readout", "state_and_readout"])
def test_general_dag_intervention_preserves_envelope_and_descendant_semantics(
    location: str,
) -> None:
    src = source()
    selected = p.sample_network_observations(
        p.NetworkObservationRequest(src, 2, profile=PROFILE, target="final"),
        rng=np.random.default_rng(7),
    )
    recipe = p.NetworkObservationRecipe(src.task, selected)
    ref = selected.catalog[selected.features[0]]
    pair = p.generate_causal_pair(
        recipe,
        interventions=(
            p.HardIntervention(
                ref.node, channels=(ref.channel,), values=(1.25,), location=location
            ),
        ),
        seed=19,
        task_id=23,
    )
    assert json.loads(pair.after.recipe_json)["recipe"]["base"] == json.loads(
        json.dumps(asdict(recipe))
    )
    assert torch.equal(pair.before.inputs, p.generate_task(recipe, seed=19, task_id=23).inputs)
    np.testing.assert_array_equal(pair.after.inputs[:, 0].numpy(), [1.25] * 16)
    original = literal(root_rows())
    expected = (
        original
        if location == "readout"
        else literal(root_rows(), (selected.features[0],), np.full((16, 1), 1.25))
    )
    np.testing.assert_allclose(pair.after.scores.numpy(), expected[:, selected.targets], atol=1e-12)
    p.explain_generation(pair.after, [15, 0, 15])
