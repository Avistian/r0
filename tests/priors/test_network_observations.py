"""Public source observation policies preserve catalog order and actual stage values."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from r0.modules import priors as p


def source(noise: bool = False) -> p.NetworkRecipe | p.NetworkInnovationRecipe:
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=4,
    )
    first = p.NetworkBlock(
        (0,), (p.NetworkAffine(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)), (0.0,) * 4),)
    )
    matrix = tuple(tuple(float(i == j) * 2 for j in range(4)) for i in range(4))
    blocks = (
        first,
        *(p.NetworkBlock((i,), (p.NetworkAffine(matrix, (float(i),) * 4),)) for i in (1, 2)),
    )
    base = p.NetworkRecipe(task, p.FrozenNetworkFunction(2, blocks))
    return (
        p.with_network_innovations(
            base,
            p.NetworkInnovationPlan((None, p.NetworkInnovationStage((1.0,) * 4, (0.2,) * 4), None)),
        )
        if noise
        else base
    )


@pytest.mark.parametrize("profile", ["p1-paper-block-r0-v1", "p1-778f0d0-v1"])
@pytest.mark.parametrize("selection", ["block", "unrestricted"])
@pytest.mark.parametrize("target", ["internal", "final"])
@pytest.mark.parametrize("order", ["preserved", "permuted"])
def test_literal_policy_indices(profile: str, selection: str, target: str, order: str) -> None:
    req = p.NetworkObservationRequest(
        source(),
        features=2,
        profile=profile,
        selection=selection,
        target=target,
        feature_order=order,
    )
    result = p.sample_network_observations(req, rng=np.random.default_rng(41))
    oracle = np.random.default_rng(41)
    n = 8 if profile.startswith("p1-778") else 12
    if profile.startswith("p1-778"):
        offset = int(oracle.integers(0, n - 3 + 1, size=1)[0]) if selection == "block" else 0
        perm = oracle.permutation(3 if selection == "block" else n - 1) + offset
        y = (n - 1,) if target == "final" else (int(perm[0]),)
        x = tuple(map(int, perm[1:3]))
    elif selection == "unrestricted":
        perm = oracle.permutation(n - int(target == "final"))
        y = (n - 1,) if target == "final" else (int(perm[0]),)
        x = tuple(map(int, perm[:2] if target == "final" else perm[1:3]))
    else:
        windows = [0, 1, 2, 4, 5, 6, 8, 9] + ([10] if target == "internal" else [])
        start = windows[int(oracle.integers(0, len(windows), size=1)[0])]
        x = (start, start + 1)
        available = [i for i in range(n) if i not in x]
        y = (n - 1,) if target == "final" else (available[int(oracle.permutation(n - 2)[0])],)
        if order == "permuted":
            x = tuple(x[int(i)] for i in oracle.permutation(2))
    if order == "preserved":
        x = tuple(sorted(x))
    assert result.features == x
    assert result.targets == y
    np.testing.assert_array_equal(result.final.restore().random(3), oracle.random(3))
    assert p.network_observations_from_dict(asdict(result)) == result


def test_real_noise_stage_catalog_and_runtime() -> None:
    src = source(True)
    assert isinstance(src, p.NetworkInnovationRecipe)
    req = p.NetworkObservationRequest(src, 2)
    result = p.sample_network_observations(req, rng=np.random.default_rng(3))
    assert tuple(c.node for c in result.catalog[4:8]) == (src.block_nodes[2],) * 4
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    graph = result.runtime(task)
    assert graph.nodes == src.nodes
    emitted = p.generate_task(graph, seed=19, task_id=2**54 + 3)
    assert p.explain_generation(emitted, [15, 0, 15])["signal_max_error"] == 0
    bad = list(result.catalog)
    bad[4] = replace(bad[4], node=bad[4].node - 1)
    with pytest.raises(ValueError, match="catalog"):
        replace(result, catalog=tuple(bad))


def test_small_catalog_rejects_and_raw_cap_is_pretransaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="insufficient"):
        p.NetworkObservationRequest(source(), 12)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("snapshot touched before raw admission")

    monkeypatch.setattr(
        type(
            p.sample_network_observations(
                p.NetworkObservationRequest(source(), 2), rng=np.random.default_rng(2)
            ).initial
        ),
        "restore",
        fail,
    )
    with pytest.raises(ValueError, match="raw"):
        p.NetworkObservationRequest(source(), 2, raw_limits=p.RawDrawLimits(numbers=1))


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
def test_supported_engines_archive_and_actual_channel_values(
    engine: type[np.random.BitGenerator],
) -> None:
    src = source()
    req = p.NetworkObservationRequest(src, 2, selection="unrestricted")
    receipt = p.sample_network_observations(req, rng=np.random.Generator(engine(21)))
    assert p.network_observations_from_dict(asdict(receipt)) == receipt
    baseline = p.generate_task(src, seed=23, task_id=2**54 + 1)
    x = baseline.inputs.numpy()
    first = np.column_stack((x[:, 0], x[:, 1], x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]))
    second = first * 2 + 1
    literal = np.concatenate((first, second, second * 2 + 2), axis=1)
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    actual = p.generate_task(receipt.runtime(task), seed=23, task_id=2**54 + 1)
    np.testing.assert_allclose(actual.inputs.numpy(), literal[:, receipt.features], atol=1e-12)
    np.testing.assert_allclose(actual.scores.numpy(), literal[:, receipt.targets], atol=1e-12)


def test_strict_archive_fields_and_primitives() -> None:
    receipt = p.sample_network_observations(
        p.NetworkObservationRequest(source(), 2), rng=np.random.default_rng(3)
    )
    for name in asdict(receipt):
        data = asdict(receipt)
        del data[name]
        with pytest.raises(ValueError, match="observation"):
            p.network_observations_from_dict(data)
    for name in ("features", "targets"):
        data = asdict(receipt)
        data[name] = tuple(float(i) for i in data[name])
        with pytest.raises(ValueError, match="observation"):
            p.network_observations_from_dict(data)
    data = asdict(receipt)
    data["raw"][0]["values"] = tuple(float(i) for i in data["raw"][0]["values"])
    with pytest.raises(ValueError, match="observation|raw"):
        p.network_observations_from_dict(data)
    with pytest.raises(ValueError, match="channel"):
        replace(receipt.catalog[0], block=False)


def test_admission_precedes_rng_and_late_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import copy

    req = p.NetworkObservationRequest(source(), 2)
    receipt = p.sample_network_observations(req, rng=np.random.default_rng(5))
    payload = asdict(receipt)
    with monkeypatch.context() as patch:

        def no_clone(*args: object, **kwargs: object) -> None:
            raise AssertionError("RNG copied before admission")

        patch.setattr(copy, "deepcopy", no_clone)
        with pytest.raises(ValueError, match="limit"):
            p.network_observations_from_dict(payload, limits=p.CausalLimits(parameters=1))
        with pytest.raises(ValueError, match="raw"):
            replace(req, raw_limits=p.RawDrawLimits(numbers=1))
    rng = np.random.default_rng(10)
    state = copy.deepcopy(rng.bit_generator.state)
    with monkeypatch.context() as patch:

        def fail_validation(self: p.SampledNetworkObservations) -> None:
            raise ValueError("injected final envelope refusal")

        patch.setattr(p.SampledNetworkObservations, "validate_replay", fail_validation)
        with pytest.raises(ValueError, match="injected"):
            p.sample_network_observations(req, rng=rng)
    assert rng.bit_generator.state == state


def test_pinned_final_target_overlap_is_exposed_without_filtering() -> None:
    req = p.NetworkObservationRequest(
        source(), 2, profile="p1-778f0d0-v1", target="final", selection="block"
    )
    # Literal pinned policy permits the forced final target inside the feature window.
    oracle = np.random.default_rng(2)
    start = int(oracle.integers(0, 6, size=1)[0])
    chosen = oracle.permutation(3) + start
    receipt = p.sample_network_observations(req, rng=np.random.default_rng(2))
    assert receipt.features == tuple(sorted(map(int, chosen[1:3])))
    assert receipt.target_feature_overlap == tuple(i for i in receipt.features if i == 7)
    assert receipt.target_feature_overlap == (7,)


def test_observation_workspace_admitted_before_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    import copy

    src = replace(source(), limits=p.CausalLimits(working_bytes=1_000_000))

    # A source can fit while the additional observation snapshots and catalog do not.
    def no_clone(*args: object, **kwargs: object) -> None:
        raise AssertionError("compiled admission happened after RNG clone")

    monkeypatch.setattr(copy, "deepcopy", no_clone)
    with pytest.raises(ValueError, match="working"):
        p.NetworkObservationRequest(src, 2)


def test_multiple_internal_classification_outputs() -> None:
    src = source()
    request = p.NetworkObservationRequest(src, 2, outputs=3, selection="unrestricted")
    receipt = p.sample_network_observations(request, rng=np.random.default_rng(18))
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=3,
    )
    actual = p.generate_task(receipt.runtime(task), seed=23, task_id=31)
    baseline = p.generate_task(src, seed=23, task_id=31)
    x = baseline.inputs.numpy()
    first = np.column_stack((x[:, 0], x[:, 1], x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]))
    second = first * 2 + 1
    literal = np.concatenate((first, second, second * 2 + 2), axis=1)
    np.testing.assert_allclose(actual.scores.numpy(), literal[:, receipt.targets], atol=1e-12)
    logits = literal[:, receipt.targets]
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    assert actual.probabilities is not None
    np.testing.assert_allclose(
        actual.probabilities.numpy(), weights / weights.sum(axis=1, keepdims=True), atol=1e-12
    )


def test_pinned_overlap_runtime_retains_values_and_receipt() -> None:
    request = p.NetworkObservationRequest(source(), 2, profile="p1-778f0d0-v1", target="final")
    receipt = p.sample_network_observations(request, rng=np.random.default_rng(2))
    before = asdict(receipt)
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    graph = receipt.runtime(task)
    assert receipt.target_feature_overlap == (7,)
    assert asdict(receipt) == before
    assert len(graph.nodes) == 5
    assert graph.target == ((4, 0),)
    assert graph.nodes[4].parents == (3,)
    actual = p.generate_task(graph, seed=19, task_id=2**54 + 13)
    np.testing.assert_array_equal(actual.inputs[:, 1].numpy(), actual.targets.numpy())
    restored = p.recipe_from_dict(asdict(graph))
    replay = p.generate_task(restored, seed=19, task_id=2**54 + 13)
    np.testing.assert_array_equal(actual.targets.numpy(), replay.targets.numpy())
    assert p.explain_generation(actual, [15, 0, 15])["signal_max_error"] == 0
    pair = p.generate_causal_pair(
        graph,
        interventions=(p.HardIntervention(3, channels=(3,), values=(7.25,)),),
        seed=19,
        task_id=2**54 + 13,
    )
    np.testing.assert_array_equal(pair.after.inputs[:, 1].numpy(), [7.25] * 16)
    np.testing.assert_array_equal(pair.after.targets.numpy(), [7.25] * 16)


def test_pinned_overlap_alias_node_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    base = replace(source(), limits=p.CausalLimits(nodes=7))
    assert isinstance(base, p.NetworkRecipe)
    stage = p.NetworkInnovationStage((1.0,) * 4, (0.2,) * 4)
    src = p.with_network_innovations(base, p.NetworkInnovationPlan((stage,) * 3))
    receipt = p.sample_network_observations(
        p.NetworkObservationRequest(src, 2, profile="p1-778f0d0-v1", target="final"),
        rng=np.random.default_rng(2),
    )
    task = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
    original = p.FrozenLinearFunction.__post_init__

    def reject_alias(function: p.FrozenLinearFunction) -> None:
        if len(function.matrix) == 1 and len(function.matrix[0]) == 4:
            raise AssertionError("alias allocated before node cap")
        original(function)

    monkeypatch.setattr(p.FrozenLinearFunction, "__post_init__", reject_alias)
    with pytest.raises(ValueError, match="alias geometry"):
        receipt.runtime(task)


def test_pinned_overlap_multiclass_alias_keeps_other_target_channels() -> None:
    receipt = p.sample_network_observations(
        p.NetworkObservationRequest(
            source(), 2, outputs=3, profile="p1-778f0d0-v1", target="final"
        ),
        rng=np.random.default_rng(2),
    )
    task = p.LinearRecipe(
        rows=16,
        context_rows=8,
        features=2,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=3,
    )
    runtime = receipt.runtime(task)
    assert receipt.features == (3, 7)
    assert receipt.targets == (5, 6, 7)
    assert runtime.target == ((3, 1), (3, 2), (4, 0))
    emitted = p.generate_task(runtime, seed=17, task_id=2**54 + 13)
    np.testing.assert_array_equal(emitted.inputs[:, 1].numpy(), emitted.scores[:, 2].numpy())
    assert emitted.probabilities is not None
    np.testing.assert_allclose(emitted.probabilities.sum(dim=1).numpy(), 1.0, rtol=0, atol=1e-14)
