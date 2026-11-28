"""Guarded controls bind actual-parent fitting to exact construction identities."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p


def test_public_population_xgboost_construction_retains_exact_inputs_and_streams():
    base = gp_root_recipe("functional").base
    controls = dict(
        topology="xgboost-population",
        base=asdict(base),
        plans=[
            dict(
                node=1,
                plan=asdict(
                    p.PopulationXGBoostPlan(
                        rows=8, input_width=2, output_width=1, estimators=2, depth=2
                    )
                ),
            )
        ],
        population="support-only",
        fit_seed=47,
        fit_task_id=59,
        engine="PCG64",
        raw_limits=asdict(p.RawDrawLimits()),
    )
    graph = p.construct_xgboost_population(
        controls, construction_seed=7, seed=47, task_id=59, limits=p.CausalLimits()
    )
    task = p.generate_task(graph, seed=47, task_id=59)
    np.testing.assert_array_equal(
        graph.fits[0].realization.inputs, task.inputs[task.context_indices]
    )
    assert graph == p.construct_xgboost_population(
        controls, construction_seed=7, seed=47, task_id=59, limits=p.CausalLimits()
    )


def test_guarded_population_construction_fits_once_and_replays_without_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import importlib
    import json

    from r0.modules import generation as g

    base = gp_root_recipe("functional").base
    controls = dict(
        topology="xgboost-population",
        base=asdict(base),
        plans=[
            dict(
                node=1,
                plan=asdict(
                    p.PopulationXGBoostPlan(
                        rows=8, input_width=2, output_width=1, estimators=2, depth=2
                    )
                ),
            )
        ],
        population="support-only",
        fit_seed=47,
        fit_task_id=59,
        engine="PCG64",
        raw_limits=asdict(p.RawDrawLimits()),
    )
    backend = importlib.import_module("xgboost")
    original = backend.train
    calls: list[int] = []

    def trained(*args: Any, **kwargs: Any):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "train", trained)
    request = g.RunRequest(
        base.task,
        seed=47,
        start=59,
        count=1,
        max_seconds=60,
        max_bytes=10_000_000,
        construction=g.GraphConstruction(json.dumps(controls), seed=7),
        causal_limits=p.CausalLimits(),
    )
    result = g.run_generation(request, tmp_path / "constructed")
    assert result.completed == 1
    assert len(calls) == 1
