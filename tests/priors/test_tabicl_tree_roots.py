"""Authored roots extend source-shaped trees without silently changing fit laws."""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def build(
    root: p.JointRootBase | p.SourceRootBase | p.PrototypeRootBase | None,
) -> p.SampledXGBoostGraph:
    return p.sample_tabicl_tree_graph(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        architecture_rng=np.random.default_rng(2),
        parameter_rng=np.random.default_rng(3),
        input_rng=np.random.default_rng(4),
        target_rng=np.random.default_rng(5),
        backend_rng=np.random.default_rng(6),
        fit_rows=32,
        root=root,
    )


@pytest.mark.parametrize(
    "root",
    [
        p.JointRootBase("uniform", 2),
        p.JointRootBase("ball", 2),
        p.SourceRootBase(
            p.SourceRootLaw(
                (
                    p.SourceRootCoordinate("unit_uniform"),
                    p.SourceRootCoordinate("categorical", probabilities=(0.2, 0.8)),
                )
            )
        ),
        p.PrototypeRootBase(p.FrozenPrototypeRoot(((-1.0, 2.0), (3.0, -4.0)))),
        p.sample_source_root_base(
            p.sample_source_root_profile(
                p.SourceRootProfile(2, selector="p1-code-normal", presample_gaussian=True),
                model_rng=np.random.default_rng(10),
                selection_rng=np.random.default_rng(11),
            ),
            modes=("standardize", "standardize"),
            calibration_rows=32,
            calibration_rng=np.random.default_rng(12),
        ),
    ],
)
def test_root_preserved_with_frozen_fits_and_archive(
    root: p.JointRootBase | p.SourceRootBase | p.PrototypeRootBase,
    tmp_path: Path,
) -> None:
    graph = build(root)
    gaussian = build(None)
    # Root selection must not consume fit streams or alter independent fit data.
    assert graph.fits == gaussian.fits
    assert graph.parameter_samples == gaussian.parameter_samples
    assert graph.nodes[0].root == root
    decoded = p.recipe_from_dict(json.loads(json.dumps(asdict(graph))))
    task = p.generate_task(decoded, seed=28, task_id=0)
    assert not torch.equal(task.inputs, p.generate_task(gaussian, seed=28, task_id=0).inputs)
    path = tmp_path / "roots.zip"
    g.save_group(path, [(0, task)], max_bytes=8000000)
    saved = next(g.load_group(path))
    assert saved.episode() == task.episode()
    assert p.explain_generation(saved, [0, 15, 0])["signal_max_error"] < 1e-12
    if type(root) is p.SourceRootBase:
        assert bool(torch.all((task.inputs[:, 1] == 0) | (task.inputs[:, 1] == 1)))


def test_invalid_root_width_does_not_consume_streams() -> None:
    streams = [np.random.default_rng(i) for i in range(5)]
    before = [copy.deepcopy(rng.bit_generator.state) for rng in streams]
    with pytest.raises(ValueError, match="invalid TabICL"):
        p.sample_tabicl_tree_graph(
            p.LinearRecipe(features=2, signal_features=2, active_features=2),
            architecture_rng=streams[0],
            parameter_rng=streams[1],
            input_rng=streams[2],
            target_rng=streams[3],
            backend_rng=streams[4],
            root=p.JointRootBase("gaussian", 3),
        )
    assert [rng.bit_generator.state for rng in streams] == before
