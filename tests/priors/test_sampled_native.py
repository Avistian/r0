"""Sampled native domains remain coupled to construction, replay and typed archives."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from r0.modules.generation import load_group, save_group
from r0.modules.priors import (
    JointFunctionPrior,
    JointRootPrior,
    NativeDomainPrior,
    NativeGraphRequest,
    generate_task,
    recipe_from_dict,
    sample_native_graph,
)


def test_sampled_native_domains_roundtrip_and_preserve_shared_scope(tmp_path: Path) -> None:
    prior = NativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(3, 4),
        emissions=("softmax",),
        frequency_laws=("dirichlet",),
    )
    request = NativeGraphRequest(
        parents=((), (0,), (1,)),
        domains=(prior,) * 3,
        observed=((0, 0), (1, 0)),
        target=(2, 0),
        rows=32,
        context_rows=20,
        calibration_rows=32,
        root_prior=JointRootPrior(
            families=("gaussian",), function=JointFunctionPrior(functions="lin")
        ),
        function_prior=JointFunctionPrior(functions="lin"),
    )
    graph = sample_native_graph(
        request,
        named_rng=np.random.default_rng(1),
        raw_rng=np.random.default_rng(2),
        calibration_rng=np.random.default_rng(3),
        fit_rng=np.random.default_rng(4),
    )
    graph.validate_replay()
    task = generate_task(graph, seed=8, task_id=9)
    assert all(feature.kind == "categorical" for feature in task.episode().features)
    assert task.targets.dtype == torch.int64
    decoded = recipe_from_dict(json.loads(json.dumps(asdict(graph))))
    assert decoded == graph
    replay = generate_task(decoded, seed=8, task_id=9)
    assert replay.mechanism_json == task.mechanism_json
    path = tmp_path / "sampled.zip"
    save_group(path, [(9, task)], max_bytes=5_000_000)
    assert next(load_group(path)).episode() == task.episode()


def _request(
    *,
    emission: str = "softmax",
    frequency: str = "dirichlet",
    threshold: str = "normal-quantile",
    function: str = "lin",
    mode: str = "meta",
) -> NativeGraphRequest:
    prior = NativeDomainPrior(
        kinds=("categorical",),
        cardinalities=(3, 4),
        emissions=(emission,),
        frequency_laws=(frequency,),
        threshold_laws=(threshold,),
    )
    functions = JointFunctionPrior(
        functions=function,
        frequencies=8,
        max_discretization_cardinality=8,
        discretization_centers=4,
    )
    return NativeGraphRequest(
        parents=((), (0,)),
        domains=(prior, prior),
        observed=((0, 0),),
        target=(1, 0),
        rows=32,
        context_rows=20,
        calibration_rows=32,
        root_prior=JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        mode=mode,
    )


def _sample(request: NativeGraphRequest):
    return sample_native_graph(
        request,
        named_rng=np.random.default_rng(1),
        raw_rng=np.random.default_rng(2),
        calibration_rng=np.random.default_rng(3),
        fit_rng=np.random.default_rng(4),
    )


def test_global_native_preferences_and_complete_raw_stream_boundaries() -> None:
    graph = _sample(_request(mode="global"))
    assert graph.domains[0].cardinality == graph.domains[1].cardinality
    assert graph.domains[0].temperature == graph.domains[1].temperature
    assert graph.domains[0].concentration == graph.domains[1].concentration
    draws = [
        d
        for d in graph.registry.decisions.categorical.draws
        if d.request.name == "native_cardinality"
    ]
    assert len(draws) == 2
    assert draws[0].preference == draws[1].preference
    for index, name in enumerate(("named", "raw", "calibration", "fit")):
        if name == "named":
            continue
        cursor = graph.initial[index].digest
        for raw in graph.raw:
            if raw.stream == name:
                assert raw.state_before == cursor
                cursor = raw.state_after
        assert cursor == graph.final[index].digest
    assert any(raw.distribution == "gamma" for raw in graph.raw)


def test_native_frequency_laws_and_threshold_adaptations() -> None:
    import pytest

    for frequency in ("uniform", "dirichlet", "logistic-normal"):
        for threshold in ("normal-quantile", "uniform-quantile", "sorted-normal"):
            graph = _sample(
                _request(emission="threshold", frequency=frequency, threshold=threshold)
            )
            graph.validate_replay()
            node = asdict(graph.recipe.nodes[1])["domain"]
            assert len(node["thresholds"]) == len(node["frequencies"]) - 1
            assert sum(node["frequencies"]) == pytest.approx(1.0)
            assert all(
                a < b for a, b in zip(node["thresholds"], node["thresholds"][1:], strict=False)
            )
            assert generate_task(graph, seed=8, task_id=9).targets.dtype == torch.int64


def test_receiving_fit_uses_one_hot_native_ancestor_population() -> None:
    graph = _sample(_request(function="tree"))
    graph.validate_replay()
    sample = np.asarray(asdict(graph)["realizations"][1]["calibration"]["sample"])
    assert sample.shape == (32, graph.domains[0].cardinality)
    assert np.all((sample == 0) | (sample == 1))
    assert np.all(sample.sum(axis=1) == 1)
    draws = [r for r in graph.raw if r.owner[:3] == ("native", "receiver:1", "calibration")]
    assert any(r.role == "uniform" and r.stream == "calibration" for r in draws)


def test_sampled_native_resplit_identity_and_tamper_rejection() -> None:
    from dataclasses import replace

    import pytest

    from r0.modules.priors import explain_generation

    graph = _sample(_request())
    task = generate_task(graph, seed=8, task_id=9)
    changed = graph.with_task(replace(graph.task, split=2, context_rows=10))
    other = generate_task(changed, seed=8, task_id=9)
    assert other.source_task_id == task.source_task_id
    assert other.episode_id != task.episode_id
    assert other.mechanism_json == task.mechanism_json
    inspected = explain_generation(task, [3, 1, 3])
    target = next(n for n in inspected["nodes"] if n["id"] == "targets")
    assert target["values"] == [[int(task.targets[i])] for i in (3, 1, 3)]
    data = json.loads(json.dumps(asdict(graph)))
    data["domains"][0]["temperature"] += 1
    with pytest.raises(ValueError, match="replay"):
        recipe_from_dict(data)


def test_native_admission_rejection_rolls_back_every_caller_stream() -> None:
    import copy
    from dataclasses import replace

    import pytest

    from r0.modules.priors import CausalLimits

    streams = [np.random.default_rng(i) for i in range(4)]
    before = [copy.deepcopy(rng.bit_generator.state) for rng in streams]
    request = replace(_request(), limits=replace(CausalLimits(), working_bytes=1_000_000))
    with pytest.raises(ValueError, match="memory"):
        sample_native_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    assert [rng.bit_generator.state for rng in streams] == before


def test_dirichlet_gamma_oracle_and_fixed_hyperparameters() -> None:
    from dataclasses import replace

    request = _request()
    prior = replace(request.domains[0], concentration=(1.0, 1.0), temperature=(1.0, 1.0))
    graph = _sample(replace(request, domains=(prior, prior)))
    for index, plan in enumerate(graph.domains):
        raw = next(r for r in graph.raw if r.owner == plan.operation and r.role == "frequency")
        assert raw.parameters == (("concentration", 1.0),)
        values = np.asarray(raw.values)
        expected = values / values.sum()
        actual = asdict(graph.recipe.nodes[index])["domain"]["frequencies"]
        np.testing.assert_array_equal(actual, expected)
    graph.validate_replay()


def test_high_cardinality_features_do_not_expand_target_classes() -> None:
    from dataclasses import replace

    request = _request(emission="threshold", frequency="uniform")
    feature = replace(request.domains[0], cardinalities=(201,))
    target = NativeDomainPrior(kinds=("continuous",))
    graph = _sample(replace(request, domains=(feature, target)))
    task = generate_task(graph, seed=7, task_id=8)
    assert task.task_type == "regression"
    assert len(task.feature_schema[0].vocabulary) == 201
    assert graph.recipe.nodes[1].equation.function.input_width == 201
    assert len(task.classes) == 0


def test_supported_generators_replay_native_domains_and_frequencies() -> None:
    request = _request(mode="local")
    for engine in (
        np.random.PCG64,
        np.random.PCG64DXSM,
        np.random.Philox,
        np.random.SFC64,
        np.random.MT19937,
    ):
        streams = [np.random.Generator(engine(i)) for i in range(4)]
        graph = sample_native_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
        graph.validate_replay()
        assert all(snapshot.engine == engine.__name__ for snapshot in graph.initial)


def test_native_frequency_underflow_fails_without_retry_or_stream_consumption() -> None:
    import copy
    from dataclasses import replace

    import pytest

    request = _request()
    prior = replace(request.domains[0], concentration=(1e-100, 1e-100))
    request = replace(request, domains=(prior, prior))
    streams = [np.random.default_rng(i) for i in range(4)]
    before = [copy.deepcopy(rng.bit_generator.state) for rng in streams]
    with pytest.raises(ValueError, match="underflowed"):
        sample_native_graph(
            request,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    assert [rng.bit_generator.state for rng in streams] == before
