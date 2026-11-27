"""Construction budgets and reduced runtime allowances are distinct on replay."""

import json
from pathlib import Path

import numpy as np
import pytest

from r0.modules import priors as p
from r0.modules.generation import load_causal_pair, save_causal_pair


@pytest.mark.parametrize("extended", [False, True])
def test_sampled_pipeline_pairs_retain_construction_limits(tmp_path: Path, extended: bool):
    prior = (
        p.JointInnovationPipelinePrior(
            channel_importance=False,
            l2_normalization=False,
            node_importance=False,
            innovation=p.JointInnovationPrior(families=("uniform",), factors=("identity",)),
        )
        if extended
        else p.JointPipelinePrior(
            channel_importance=False,
            l2_normalization=False,
            node_importance=False,
            gaussian_noise=True,
        )
    )
    functions = p.JointFunctionPrior(functions="lin")
    request = p.GeometryPipelineRequest(
        topology=p.AttachmentGraphPrior(nodes=2),
        task=p.LinearRecipe(
            rows=16, context_rows=8, features=1, signal_features=1, active_features=1
        ),
        width_prior=p.JointWidthPrior(low=1, high=2),
        calibration_rows=8,
        root_prior=p.JointRootPrior(families=("gaussian",), function=functions),
        function_prior=functions,
        pipeline_prior=prior,
    )
    graph = p.sample_geometry_graph(
        request,
        **{
            key: np.random.default_rng(i)
            for i, key in enumerate(("named_rng", "raw_rng", "calibration_rng", "fit_rng"))
        },
    )
    assert graph.runtime.limits.parameters < request.limits.parameters
    pair = p.generate_causal_pair(
        graph,
        interventions=(p.StageIntervention(0, replace_pipeline=True),),
        seed=5,
        task_id=9,
    )
    path = tmp_path / "pair"
    save_causal_pair(path, pair, max_bytes=10_000_000)
    restored = load_causal_pair(path)
    assert json.loads(restored.card_json) == json.loads(pair.card_json)
    # A caller's explicit smaller budget still applies to the entire retained archive.
    with pytest.raises(ValueError, match="caller limits"):
        p.compare_causal_pair(pair.before, pair.after, limits=graph.runtime.limits)
