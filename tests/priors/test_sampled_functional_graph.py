"""Sampled functional graphs preserve complete construction and independent fits."""

import copy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from r0.modules import priors


@pytest.mark.parametrize("family", ["lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"])
def test_sampled_graph_replays_plans_realizations_populations_and_all_four_streams(
    family: str,
) -> None:
    functions = priors.JointFunctionPrior(
        functions=family,
        frequencies=4,
        matrix=priors.JointMatrixPrior(families=("gaussian",)),
    )
    request = priors.JointGraphRequest(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        parents=((), (0,), (0, 1)),
        widths=(1, 2, 1),
        observed=((0, 0),),
        target=((2, 0),),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
    )
    streams = {
        name: np.random.default_rng(seed)
        for name, seed in zip(
            ("named", "raw", "calibration", "fit"), (114, 115, 116, 117), strict=True
        )
    }
    replay_streams = copy.deepcopy(streams)
    graph = priors.sample_joint_graph(
        request, **{f"{name}_rng": rng for name, rng in streams.items()}
    )
    graph.validate()
    repeated = priors.sample_joint_graph(
        request, **{f"{name}_rng": rng for name, rng in replay_streams.items()}
    )
    assert asdict(graph) == asdict(repeated)
    assert asdict(graph.replay()) == asdict(graph)
    for name in streams:
        np.testing.assert_array_equal(streams[name].random(8), replay_streams[name].random(8))
    if family in ("tree", "disc", "em", "mlp"):
        assert [receipt.node for receipt in graph.calibrations] == [1, 2]
        assert graph.calibrations[0].raw_stop <= graph.calibrations[1].raw_start
        assert graph.calibrations[0].identity != graph.calibrations[1].identity
    assert graph.runtime.nodes[1].function.output_width == 2
    assert graph.runtime.nodes[2].function.input_width == 3
    with pytest.raises(ValueError, match="runtime"):
        replace(graph, runtime=replace(graph.runtime, observed=((1, 0),))).validate()


def test_sampled_graph_late_budget_failure_preserves_all_four_caller_generators() -> None:
    functions = priors.JointFunctionPrior(
        functions="tree", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    request = priors.JointGraphRequest(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        parents=((), (0,)),
        widths=(1, 1),
        observed=((0, 0),),
        target=((1, 0),),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        raw_limits=priors.RawDrawLimits(draws=2),
    )
    streams = {
        name: np.random.default_rng(seed)
        for name, seed in zip(
            ("named", "raw", "calibration", "fit"), (114, 115, 116, 117), strict=True
        )
    }
    controls = copy.deepcopy(streams)
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.sample_joint_graph(request, **{f"{name}_rng": rng for name, rng in streams.items()})
    for name in streams:
        np.testing.assert_array_equal(streams[name].random(8), controls[name].random(8))


def _small_request(*, root: str = "gaussian", family: str = "lin") -> priors.JointGraphRequest:
    linear = priors.JointFunctionPrior(
        functions="lin,tree" if family == "tree" else "lin",
        matrix=priors.JointMatrixPrior(families=("gaussian",)),
    )
    return priors.JointGraphRequest(
        task=priors.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        parents=((), (), (1, 0)),
        widths=(1, 1, 1),
        observed=((0, 0),),
        target=((2, 0),),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=(root,), function=linear),
        function_prior=linear,
    )


def _sample(
    request: priors.JointGraphRequest, engine: str = "PCG64"
) -> priors.SampledFunctionalGraph:
    streams = [
        np.random.Generator(getattr(np.random, engine)(seed)) for seed in (13, 115, 116, 117)
    ]
    for rng in streams:
        # Retain cached uint32/normal buffers and noninitial counters as well as seeds.
        rng.integers(10, dtype=np.uint32)
        rng.normal(size=3)
    return priors.sample_joint_graph(
        request,
        named_rng=streams[0],
        raw_rng=streams[1],
        calibration_rng=streams[2],
        fit_rng=streams[3],
    )


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
@pytest.mark.parametrize("root", ["gaussian", "uniform", "ball", "covariance"])
def test_graph_replay_restores_noninitial_buffered_states_for_each_supported_engine(
    engine: str, root: str
) -> None:
    graph = _sample(_small_request(root=root), engine)
    assert graph.replay() == graph


@pytest.mark.parametrize("with_pipeline_and_noise", [False, True])
def test_fitted_child_uses_fresh_root_draws_in_declared_parent_order(
    with_pipeline_and_noise: bool,
) -> None:
    request = _small_request(family="tree")
    if with_pipeline_and_noise:
        request = replace(
            request,
            authored_pipelines=(
                priors.CausalNodePipeline(
                    1,
                    channel_importance=(3.0,),
                    node_importance=5.0,
                    converters=(priors.NumericalConverter(0),),
                ),
                None,
                None,
            ),
            noise=(
                priors.JointGraphNoise((2.0,), response=priors.NoiseResponse(strength=2.0)),
                priors.JointGraphNoise(),
                priors.JointGraphNoise(),
            ),
        )
    graph = _sample(request)
    (receipt,) = graph.calibrations
    draws = graph.raw[receipt.raw_start : receipt.raw_stop]
    inputs: list[NDArray[np.float64]] = []
    for parent in (1, 0):
        (draw,) = [
            r
            for r in draws
            if r.owner[-2:] == (f"node:{parent}", "root") and r.role == "root-base-values"
        ]
        function = graph.runtime.nodes[parent].function
        assert isinstance(function, priors.FrozenLinearFunction)
        expected = np.asarray(draw.values, dtype=np.float64) * function.matrix[0][0]
        if with_pipeline_and_noise and parent == 0:
            (innovation,) = [
                r
                for r in draws
                if r.owner[-3:] == ("node:0", "innovation", "column:0") and r.role == "values"
            ]
            expected = (
                3 * expected + 2 * (1 + 2 * np.abs(expected)) * np.asarray(innovation.values)
            ) * 5
        inputs.append(expected)
        root = graph.realizations[parent]
        assert isinstance(root, priors.JointRootRealization)
        assert tuple(draw.values) != tuple(row[0] for row in root.calibration.values)
    child = graph.realizations[2]
    assert isinstance(child, priors.JointTreeFunctionRealization)
    np.testing.assert_allclose(child.calibration.sample, np.column_stack(inputs), rtol=0, atol=0)
    assert receipt.parents == (1, 0)


@pytest.mark.parametrize(
    "mutation",
    ["raw-role", "raw-value", "receipt", "initial", "final", "runtime-equation", "owner"],
)
def test_graph_validation_binds_raw_provenance_receipts_and_runtime(mutation: str) -> None:
    graph = _sample(_small_request(family="tree"))
    if mutation == "raw-role":
        graph = replace(graph, raw=(replace(graph.raw[0], role="foreign-role"), *graph.raw[1:]))
    elif mutation == "raw-value":
        first = graph.raw[0]
        graph = replace(
            graph,
            raw=(
                replace(first, values=(float(first.values[0]) + 1, *first.values[1:])),
                *graph.raw[1:],
            ),
        )
    elif mutation == "receipt":
        graph = replace(graph, calibrations=(replace(graph.calibrations[0], raw_start=0),))
    elif mutation == "initial":
        graph = replace(
            graph, initial=(replace(graph.initial[0], payload="{}"), *graph.initial[1:])
        )
    elif mutation == "final":
        graph = replace(graph, final=graph.initial)
    elif mutation == "runtime-equation":
        first = graph.runtime.nodes[0]
        changed = replace(first, function=priors.FrozenLinearFunction(((42.0,),)))
        graph = replace(
            graph, runtime=replace(graph.runtime, nodes=(changed, *graph.runtime.nodes[1:]))
        )
    else:
        graph = replace(graph, owners=graph.owners[:-1])
    with pytest.raises(ValueError, match="replay|snapshot|ownership|runtime|receipt"):
        graph.validate()


def test_graph_emission_row_count_does_not_change_fitted_constants() -> None:
    request = _small_request(family="tree")
    first = _sample(request)
    second = _sample(replace(request, task=replace(request.task, rows=24)))
    assert first.plan == second.plan
    assert first.realizations == second.realizations
    assert first.calibrations == second.calibrations
    assert first.raw == second.raw
    assert first.final == second.final


@pytest.mark.parametrize("role", [0, 1, 2, 3])
def test_graph_rejects_overridden_engine_setters_before_any_caller_changes(role: int) -> None:
    class PCG64(np.random.PCG64):
        @property
        def state(self) -> Any:
            return super().state

        @state.setter
        def state(self, value: Any) -> None:
            raise RuntimeError("injected publication failure")

        def __deepcopy__(self, memo: dict[int, Any]) -> np.random.PCG64:
            clone = np.random.PCG64(0)
            clone.state = self.state
            return clone

    streams = [np.random.default_rng(seed) for seed in (13, 115, 116, 117)]
    streams[role] = np.random.Generator(PCG64(115))
    before = [copy.deepcopy(rng.bit_generator.state) for rng in streams]
    with pytest.raises(ValueError, match="supported.*generator"):
        priors.sample_joint_graph(
            _small_request(),
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    assert [rng.bit_generator.state for rng in streams] == before
