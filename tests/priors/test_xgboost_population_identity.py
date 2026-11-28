"""Independent ancestor fits cannot silently become ordinary fitting-row queries."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import generation as g
from r0.modules import priors as p


@pytest.fixture(scope="module")
def independent():
    return p.fit_xgboost_population_graph(
        gp_root_recipe("functional").base,
        (
            (
                1,
                p.PopulationXGBoostPlan(
                    rows=16, input_width=2, output_width=1, estimators=2, depth=2
                ),
            ),
        ),
        population="independent-ancestors",
        seed=47,
        task_id=59,
        target_rng=np.random.default_rng(7),
        backend_rng=np.random.default_rng(11),
    )


@pytest.mark.parametrize("entry", ["generate", "typed-run", "raw-run", "range", "inspection"])
def test_independent_fit_identity_rejected_before_restore(
    independent: p.XGBoostPopulationGraph, monkeypatch: pytest.MonkeyPatch, entry: str
):
    task = p.generate_task(independent, seed=48, task_id=59)

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("fitting identity initiated replay")

    monkeypatch.setattr(type(independent.fits[0].realization.initial_rng[0]), "restore", forbidden)

    def invoke():
        if entry == "generate":
            p.generate_task(independent, seed=47, task_id=59)
        elif entry == "typed-run":
            g.RunRequest(
                independent, seed=47, start=59, count=1, max_seconds=60, max_bytes=10_000_000
            )
        elif entry in ("raw-run", "range"):
            g.request_from_dict(
                dict(
                    recipe=asdict(independent),
                    seed=47,
                    start=58 if entry == "range" else 59,
                    count=2 if entry == "range" else 1,
                    max_seconds=60,
                    max_bytes=10_000_000,
                )
            )
        else:
            record = json.loads(task.recipe_json)
            record["seed"] = 47
            p.explain_generation(replace(task, recipe_json=json.dumps(record)), [0])

    with pytest.raises(ValueError, match="independent.*identity"):
        invoke()


def test_independent_query_identity_preserves_frozen_fit(independent: p.XGBoostPopulationGraph):
    task = p.generate_task(independent, seed=48, task_id=59)
    np.testing.assert_array_equal(
        task.targets,
        p.evaluate_xgboost_vector(independent.fits[0].realization.function, task.inputs.numpy())[
            :, 0
        ],
    )
    assert p.validate_xgboost_population_task(task, limits=p.CausalLimits()) == independent
