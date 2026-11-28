"""Whole construction transactions commit named choices and raw streams together."""

import pickle

import numpy as np
import pytest

from r0.modules import priors


def _root(
    context: priors.JointSamplingContext, name: str, rows: int = 4
) -> priors.JointRootRealization:
    plan = priors.sample_joint_root(
        context,
        name,
        width=2,
        calibration_rows=rows,
        prior=priors.JointRootPrior(
            families=("covariance",),
            function=priors.JointFunctionPrior(
                functions="tree",
                matrix=priors.JointMatrixPrior(families=("gaussian",), vector_family="uniform"),
            ),
        ),
    )
    return priors.materialize_joint_root(
        plan,
        context.freeze(),
        rng=context.stream("construction"),
        calibration_rng=context.stream("calibration"),
        fit_rng=context.stream("fit"),
        calibration_id=f"{name}-calibration",
    )


def _states(generators: list[np.random.Generator]) -> bytes:
    return pickle.dumps([generator.bit_generator.state for generator in generators])


@pytest.mark.parametrize("mode", ["local", "global", "meta"])
@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_failed_graph_after_real_root_fit_rolls_back_named_registry_and_all_raw_streams(
    mode: str,
    engine: type[np.random.BitGenerator],
) -> None:
    streams = [np.random.Generator(engine(seed)) for seed in (110, 111, 112, 113)]
    context = priors.JointSamplingContext(
        streams[0],
        mode=mode,
        streams=dict(zip(("construction", "calibration", "fit"), streams[1:], strict=True)),
    )
    before = context.freeze_registry()
    state = _states(streams)
    with (  # noqa: PT012
        pytest.raises(ValueError, match="later graph failure"),
        context.transaction("graph") as graph,
    ):
        root = _root(graph, "first")
        assert len(root.outputs) == 4
        assert _states(
            [graph.stream(name) for name in ("construction", "calibration", "fit")]
        ) != _states(streams[1:])
        _root(graph, "second")
        raise ValueError("later graph failure")
    assert context.freeze_registry() == before
    assert _states(streams) == state
    with context.transaction("graph") as graph:
        recovered = (_root(graph, "first"), _root(graph, "second"))
    fresh_streams = [np.random.Generator(engine(seed)) for seed in (110, 111, 112, 113)]
    fresh = priors.JointSamplingContext(
        fresh_streams[0],
        mode=mode,
        streams=dict(zip(("construction", "calibration", "fit"), fresh_streams[1:], strict=True)),
    )
    with fresh.transaction("graph") as graph:
        expected = (_root(graph, "first"), _root(graph, "second"))
    assert recovered == expected
    assert context.freeze_registry() == fresh.freeze_registry()
    assert _states(streams) == _states(fresh_streams)
    assert _states(streams) != state


@pytest.mark.parametrize(("left", "right"), [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
def test_joint_streams_reject_shared_bit_generators_before_any_draw(left: int, right: int) -> None:
    streams = [np.random.default_rng(seed) for seed in range(4)]
    streams[right] = np.random.Generator(streams[left].bit_generator)
    before = _states(streams)
    with pytest.raises(ValueError, match="distinct bit generators"):
        priors.JointSamplingContext(
            streams[0], streams=dict(zip(("raw", "calibration", "fit"), streams[1:], strict=True))
        )
    assert _states(streams) == before


@pytest.mark.parametrize("changed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937, np.random.Philox, np.random.SFC64],
)
def test_external_mutation_rejects_entire_joint_commit_but_preserves_the_external_draw(
    changed: int,
    engine: type[np.random.BitGenerator],
) -> None:
    streams = [np.random.Generator(engine(seed)) for seed in (110, 111, 112, 113)]
    controls = [np.random.Generator(engine(seed)) for seed in (110, 111, 112, 113)]
    context = priors.JointSamplingContext(
        streams[0],
        streams=dict(zip(("construction", "calibration", "fit"), streams[1:], strict=True)),
    )
    before = context.freeze_registry()
    with (  # noqa: PT012
        pytest.raises(ValueError, match="caller RNG changed"),
        context.transaction("graph") as graph,
    ):
        _root(graph, "root")
        streams[changed].random()
    controls[changed].random()
    assert _states(streams) == _states(controls)
    assert context.freeze_registry() == before


def test_stream_handles_close_and_nested_commit_does_not_alias_stale_child_generator() -> None:
    named, raw = np.random.default_rng(110), np.random.default_rng(111)
    context = priors.JointSamplingContext(named, streams={"raw": raw})
    with pytest.raises(ValueError, match="inside an operation"):
        context.stream("raw")
    with context.transaction("graph") as graph:
        parent_rng = graph.stream("raw")
        with graph.transaction("child") as child:
            with pytest.raises(ValueError, match="active child"):
                graph.stream("raw")
            with pytest.raises(ValueError, match="not registered"):
                child.stream("missing")
            child_rng = child.stream("raw")
            actual = child_rng.random(4)
        with pytest.raises(ValueError, match="closed"):
            child.stream("raw")
        parent_state = _states([parent_rng])
        child_rng.random(100)
        assert _states([parent_rng]) == parent_state
        assert graph.stream("raw") is parent_rng
    with pytest.raises(ValueError, match="closed"):
        graph.stream("raw")
    oracle = np.random.default_rng(111)
    np.testing.assert_array_equal(actual, oracle.random(4))
    assert _states([raw]) == _states([oracle])
    graph_state = _states([raw])
    parent_rng.random(100)
    assert _states([raw]) == graph_state


def test_caught_child_failure_preserves_prior_parent_work_and_allows_child_retry() -> None:
    named, raw = np.random.default_rng(110), np.random.default_rng(111)
    context = priors.JointSamplingContext(named, streams={"raw": raw}, mode="local")
    with context.transaction("graph") as graph:
        first = graph.stream("raw").random(3)
        before = graph.freeze()
        states = _states([graph.stream("raw")])
        with pytest.raises(ValueError, match="child failed"), graph.transaction("child") as child:  # noqa: PT012
            child.numerical("scale", priors.NumericalParameter("scale", 0.0, 1.0))
            child.stream("raw").normal(size=10)
            raise ValueError("child failed")
        assert graph.freeze() == before
        assert _states([graph.stream("raw")]) == states
        with graph.transaction("child") as child:
            child.numerical("scale", priors.NumericalParameter("scale", 0.0, 1.0))
            second = child.stream("raw").normal(size=10)
    oracle = np.random.default_rng(111)
    np.testing.assert_array_equal(first, oracle.random(3))
    np.testing.assert_array_equal(second, oracle.normal(size=10))
    assert _states([raw]) == _states([oracle])
    assert [span.operation for span in context.freeze_registry().operations] == [
        ("graph",),
        ("graph", "child"),
    ]


def test_stream_registration_copies_mapping_and_admits_count_and_state_bytes() -> None:
    named, raw, alternate = (np.random.default_rng(seed) for seed in (110, 111, 112))
    streams = {"raw": raw}
    context = priors.JointSamplingContext(named, streams=streams)
    streams["raw"] = alternate
    with context.transaction("graph") as graph:
        result = graph.stream("raw").random()
    oracle = np.random.default_rng(111)
    assert result == oracle.random()
    assert _states([raw]) == _states([oracle])
    assert _states([alternate]) == _states([np.random.default_rng(112)])
    before = _states([named, raw, alternate])
    with pytest.raises(ValueError, match="stream count limit"):
        priors.JointSamplingContext(
            named,
            streams={"raw": raw, "fit": alternate},
            stream_limits=priors.JointStreamLimits(count=1),
        )
    with pytest.raises(ValueError, match="stream state byte limit"):
        priors.JointSamplingContext(
            named, streams={"raw": raw}, stream_limits=priors.JointStreamLimits(state_bytes=1)
        )
    assert _states([named, raw, alternate]) == before


def test_later_root_admission_failure_rolls_back_an_already_fitted_root() -> None:
    generators = [np.random.default_rng(seed) for seed in (110, 111, 112, 113)]
    context = priors.JointSamplingContext(
        generators[0],
        streams=dict(zip(("construction", "calibration", "fit"), generators[1:], strict=True)),
    )
    before, state = context.freeze_registry(), _states(generators)
    with pytest.raises(ValueError, match="limit"), context.transaction("graph") as graph:  # noqa: PT012
        completed = _root(graph, "first")
        assert len(completed.outputs) == 4
        _root(graph, "oversized", rows=10**9)
    assert context.freeze_registry() == before
    assert _states(generators) == state
