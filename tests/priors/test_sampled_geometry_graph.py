"""Complete geometry archives keep one preference scope and all construction evidence."""

import copy
import json
from dataclasses import asdict, replace
from typing import cast

import numpy as np
import pytest

from r0.modules import priors
from r0.modules.episodes import TaskType


def _request(
    task_type: str = "regression", *, subsample: bool = True
) -> priors.GeometryGraphRequest:
    functions = priors.JointFunctionPrior(
        functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    classes = {"regression": 0, "binary": 2, "multiclass": 3}[task_type]
    return priors.GeometryGraphRequest(
        topology=priors.CauchyGraphPrior(nodes=3, offset=1000.0),
        task=priors.LinearRecipe(
            rows=16,
            context_rows=8,
            features=2,
            signal_features=2,
            active_features=2,
            task_type=cast(TaskType, task_type),
            classes=classes,
        ),
        subsample=subsample,
        width_prior=priors.JointWidthPrior(low=1, high=3),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
    )


def _sample(request: priors.GeometryGraphRequest) -> priors.SampledGeometryGraph:
    return priors.sample_geometry_graph(
        request,
        named_rng=np.random.default_rng(114),
        raw_rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
    )


@pytest.mark.parametrize("task_type", ["regression", "binary", "multiclass"])
@pytest.mark.parametrize("subsample", [False, True])
def test_geometry_archive_generation_inspection_and_source_identity(
    task_type: str, subsample: bool
) -> None:
    result = _sample(_request(task_type, subsample=subsample))
    assert result.replay() == result
    archive = json.loads(json.dumps(asdict(result)))
    untouched = copy.deepcopy(archive)
    assert priors.recipe_from_dict(archive) == result
    assert archive == untouched
    emitted = priors.generate_task(result, seed=19, task_id=7)
    assert priors.recipe_from_dict(json.loads(emitted.recipe_json)["recipe"]) == result
    assert priors.explain_generation(emitted, [7, 0, 7])["rows"] == [7, 0, 7]
    resplit = result.with_task(replace(result.task, context_rows=4, split=1))
    other = priors.generate_task(resplit, seed=19, task_id=7)
    assert emitted.source_task_id == other.source_task_id
    assert emitted.episode_id != other.episode_id
    np.testing.assert_array_equal(emitted.inputs.numpy(), other.inputs.numpy())
    np.testing.assert_array_equal(emitted.targets.numpy(), other.targets.numpy())
    change = "noise" if task_type == "regression" else "separation"
    changed = priors.paired_recipe(result, intervention=change, value=0.25)
    assert isinstance(changed, priors.SampledGeometryGraph)
    assert changed.observations == result.observations
    assert changed.widths == result.widths
    assert changed.graph.registry == result.graph.registry
    assert changed.graph.raw == result.graph.raw
    changed_task = priors.generate_task(changed, seed=19, task_id=7)
    assert changed_task.source_task_id != emitted.source_task_id


def test_geometry_mechanisms_use_the_existing_named_scope() -> None:
    result = _sample(_request())
    named = result.initial[0].restore()
    context = priors.JointSamplingContext(named, mode=result.request.mode)
    observation_plan = priors.sample_joint_observations(
        context,
        "observations",
        nodes=3,
        groups=result.request.groups,
        subsample=result.request.subsample,
        limits=result.request.limits,
    )
    assert observation_plan == result.observations.plan
    widths = priors.sample_joint_widths(
        context,
        "widths",
        converter_channels=result.layout.converter_channels,
        prior=result.request.width_prior,
        limits=result.request.limits,
    )
    assert widths == result.widths
    plan = priors.sample_joint_graph_plan(
        context,
        "graph",
        parents=result.topology.parents,
        widths=widths.widths,
        calibration_rows=4,
        root_prior=result.request.root_prior,
        function_prior=result.request.function_prior,
        limits=result.graph.request.limits,
    )
    assert plan == result.graph.plan
    with pytest.raises(ValueError, match="cannot become explicit graph v1"):
        result.graph.explicit_graph(result.initial, result.final)
    assert context.freeze_registry() == result.graph.registry
    np.testing.assert_array_equal(named.random(8), result.final[0].restore().random(8))
    assert result.runtime.nodes[2].function.input_width == sum(widths.widths[:2])
    assert all(
        width > converters
        for width, converters in zip(widths.widths, result.layout.converter_channels, strict=True)
    )


def test_geometry_tampering_limits_and_four_stream_rollback() -> None:
    request = _request()
    result = _sample(request)
    raw = json.loads(json.dumps(asdict(result)))
    for section in ("topology", "observations", "layout", "widths", "graph", "initial", "final"):
        broken = copy.deepcopy(raw)
        del broken[section]
        with pytest.raises(ValueError, match="invalid geometry|differs from complete replay"):
            priors.geometry_graph_from_dict(broken)
    first = next(i for i, record in enumerate(result.graph.raw) if record.stream == "raw")
    changed_raw = list(result.graph.raw)
    changed_raw[first] = replace(changed_raw[first], state_before="0" * 64)
    changed_graph = replace(result.graph, raw=tuple(changed_raw))
    with pytest.raises(ValueError, match="raw stream boundaries differ"):
        replace(result, graph=changed_graph).validate_structure()
    wrong_final = (result.final[0], result.initial[1], *result.final[2:])
    with pytest.raises(ValueError, match="final stream boundary differs"):
        replace(result, final=wrong_final).validate_structure()
    broken = copy.deepcopy(raw)
    broken["widths"]["latent"][0] += 1
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.geometry_graph_from_dict(broken)
    limited = replace(request, raw_limits=priors.RawDrawLimits(draws=len(result.graph.raw)))
    streams = [snapshot.restore() for snapshot in result.initial]
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.sample_geometry_graph(
            limited,
            named_rng=streams[0],
            raw_rng=streams[1],
            calibration_rng=streams[2],
            fit_rng=streams[3],
        )
    for rng, snapshot in zip(streams, result.initial, strict=True):
        np.testing.assert_array_equal(rng.random(8), snapshot.restore().random(8))
    larger = _sample(replace(request, limits=priors.CausalLimits(parameters=1_000_001)))
    archive = json.loads(json.dumps(asdict(larger)))
    assert priors.recipe_from_dict(archive, limits=larger.request.limits) == larger
    with pytest.raises(ValueError, match="declared limits exceed caller limits"):
        priors.recipe_from_dict(archive)
    pair = priors.generate_pair(
        larger, intervention="noise", value=0.25, seed=19, task_id=7, limits=larger.request.limits
    )
    assert all(json.loads(pair.card_json)["required_equalities"].values())
    emitted = priors.generate_task(larger, seed=19, task_id=7)
    assert priors.explain_generation(emitted, [0], limits=larger.request.limits)["rows"] == [0]
    with pytest.raises(ValueError, match="declared limits exceed caller limits"):
        priors.explain_generation(emitted, [0])


@pytest.mark.parametrize(
    ("engine", "family", "mode"),
    [
        ("PCG64", "tree", "local"),
        ("PCG64DXSM", "mlp", "global"),
        ("Philox", "prod", "meta"),
        ("SFC64", "em", "local"),
        ("MT19937", "disc", "global"),
    ],
)
def test_geometry_fitted_families_advanced_engines_and_registry_closure(
    engine: str, family: str, mode: str
) -> None:
    functions = priors.JointFunctionPrior(
        functions=family, frequencies=4, matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    request = replace(
        _request(),
        mode=mode,
        categorical_policy="refresh",
        function_prior=functions,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
    )
    streams = [
        np.random.Generator(getattr(np.random, engine)(seed)) for seed in (114, 115, 116, 117)
    ]
    for stream in streams:
        stream.random(3)
    initial = copy.deepcopy(streams)
    if family == "mlp":
        # This retained source draw selects calibrated exp. The existing moment
        # contract rejects uncertified second moments, without resampling.
        with pytest.raises(ValueError, match="certified finite output second moments"):
            priors.sample_geometry_graph(
                request,
                named_rng=streams[0],
                raw_rng=streams[1],
                calibration_rng=streams[2],
                fit_rng=streams[3],
            )
        for rng, control in zip(streams, initial, strict=True):
            np.testing.assert_array_equal(rng.random(8), control.random(8))
        return
    result = priors.sample_geometry_graph(
        request,
        named_rng=streams[0],
        raw_rng=streams[1],
        calibration_rng=streams[2],
        fit_rng=streams[3],
    )
    assert result.replay() == result
    assert priors.recipe_from_dict(json.loads(json.dumps(asdict(result)))) == result
    repeated = priors.sample_geometry_graph(
        request,
        named_rng=initial[0],
        raw_rng=initial[1],
        calibration_rng=initial[2],
        fit_rng=initial[3],
    )
    assert repeated == result
    for rng, control in zip(streams, initial, strict=True):
        np.testing.assert_array_equal(rng.random(8), control.random(8))
    registry = result.graph.registry
    foreign = priors.JointOperationSpan(
        ("unowned",), len(registry.decisions.events), len(registry.decisions.events)
    )
    changed_registry = replace(registry, operations=(*registry.operations, foreign))
    changed = replace(result, graph=replace(result.graph, registry=changed_registry))
    with pytest.raises(ValueError, match="foreign owners"):
        changed.validate_structure()
