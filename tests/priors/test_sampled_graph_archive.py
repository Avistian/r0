"""The shared recipe archive retains sampled construction rather than authored provenance."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from typing import cast

import numpy as np
import pytest

from r0.modules import priors
from r0.modules.episodes import TaskType


@pytest.mark.parametrize("family", ["lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"])
@pytest.mark.parametrize("root", ["gaussian", "uniform", "ball", "covariance"])
def test_sampled_graph_round_trips_through_shared_recipe_decoder(family: str, root: str) -> None:
    functions = priors.JointFunctionPrior(
        functions=family, frequencies=4, matrix=priors.JointMatrixPrior(families=("gaussian",))
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
        root_prior=priors.JointRootPrior(families=(root,), function=functions),
        function_prior=functions,
    )
    graph = priors.sample_joint_graph(
        request,
        named_rng=np.random.default_rng(114),
        raw_rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
    )
    raw = json.loads(json.dumps(asdict(graph), allow_nan=False))
    restored = priors.recipe_from_dict(raw)
    assert isinstance(restored, priors.SampledFunctionalGraph)
    assert restored == graph
    assert restored.replay() == graph
    emitted = priors.generate_task(restored, seed=19, task_id=7)
    resplit = restored.with_task(replace(restored.task, context_rows=4, split=1))
    another_episode = priors.generate_task(resplit, seed=19, task_id=7)
    assert emitted.source_task_id == another_episode.source_task_id
    assert emitted.episode_id != another_episode.episode_id
    np.testing.assert_array_equal(emitted.inputs.numpy(), another_episode.inputs.numpy())
    np.testing.assert_array_equal(emitted.targets.numpy(), another_episode.targets.numpy())
    card = json.loads(emitted.mechanism_json)
    assert card["sampling_provenance"] == "retained-four-stream-construction-v1"
    assert card["construction"] == "sampled-frozen-functions-v1"
    saved = json.loads(emitted.recipe_json)
    assert priors.recipe_from_dict(saved["recipe"]) == graph
    explanation = priors.explain_generation(emitted, [7, 0, 7])
    assert explanation["rows"] == [7, 0, 7]
    assert explanation["nodes"]
    corrupted = deepcopy(raw)
    corrupted["raw"][0]["role"] = "altered-role"
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.sampled_graph_from_dict(corrupted)
    missing = deepcopy(raw)
    del missing["final"]
    with pytest.raises(ValueError, match="differs from complete replay"):
        priors.sampled_graph_from_dict(missing)
    with pytest.raises(ValueError, match="limit"):
        priors.sampled_graph_from_dict(raw, limits=priors.CausalLimits(parameters=10))


@pytest.mark.parametrize(
    ("task_type", "classes"), [("regression", 0), ("binary", 2), ("multiclass", 3)]
)
def test_sampled_archive_preserves_pipeline_noise_and_paired_task_controls(
    task_type: str, classes: int
) -> None:
    functions = priors.JointFunctionPrior(
        functions="lin", matrix=priors.JointMatrixPrior(families=("gaussian",))
    )
    outputs = classes if task_type == "multiclass" else 1
    request = priors.JointGraphRequest(
        task=priors.LinearRecipe(
            rows=16,
            context_rows=8,
            features=1,
            signal_features=1,
            active_features=1,
            task_type=cast(TaskType, task_type),
            classes=classes,
        ),
        parents=((), (0,)),
        widths=(1, outputs),
        observed=((0, 0),),
        target=tuple((1, i) for i in range(outputs)),
        calibration_rows=4,
        root_prior=priors.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        authored_pipelines=(
            priors.CausalNodePipeline(1, channel_importance=(2.0,), node_importance=3.0),
            None,
        ),
        noise=(
            priors.JointGraphNoise((0.3,)),
            priors.JointGraphNoise(tuple(0.1 for _ in range(outputs))),
        ),
    )
    graph = priors.sample_joint_graph(
        request,
        named_rng=np.random.default_rng(114),
        raw_rng=np.random.default_rng(115),
        calibration_rng=np.random.default_rng(116),
        fit_rng=np.random.default_rng(117),
    )
    restored = priors.sampled_graph_from_dict(json.loads(json.dumps(asdict(graph))))
    assert restored == graph
    intervention = "noise" if task_type == "regression" else "separation"
    changed = priors.paired_recipe(restored, intervention=intervention, value=0.25)
    assert isinstance(changed, priors.SampledFunctionalGraph)
    assert changed.realizations == restored.realizations
    assert changed.raw == restored.raw
    assert changed.final == restored.final
    assert (
        getattr(changed.task, "noise_ratio" if task_type == "regression" else "separation") == 0.25
    )
    if task_type == "regression":
        pair = priors.generate_pair(restored, intervention="noise", value=0.25, seed=19, task_id=7)
        assert all(json.loads(pair.card_json)["required_equalities"].values())
    generated = priors.generate_task(restored, seed=19, task_id=7)
    changed_generated = priors.generate_task(changed, seed=19, task_id=7)
    assert generated.source_task_id != changed_generated.source_task_id
    assert generated.task_type == task_type
    assert priors.explain_generation(generated, [2, 0])["rows"] == [2, 0]
