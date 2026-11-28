"""Malformed GP comparison custody rejects before any covariance or RNG work."""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p


@pytest.fixture(scope="module")
def archive():
    base = gp_root_recipe("functional")
    change = p.CausalInterventionRecipe(base.base, (p.HardIntervention(1, (0,), (2.0,)),))
    return asdict(p.intervene_gp_root_dag(base, change))


@pytest.mark.parametrize(
    "bad", ["version", "identity", "draw", "graph", "extra", "geometry", "caps"]
)
def test_bad_gp_intervention_metadata_has_zero_replay(
    archive: dict[str, Any],
    bad: str,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = deepcopy(archive)
    limits = p.CausalLimits()
    if bad == "version":
        raw["version"] = "unknown"
    elif bad == "identity":
        raw["population"]["seed"] += 1
    elif bad == "draw":
        rows = [list(row) for row in raw["population"]["draw"]["values"]]
        rows[0][0] += 1
        raw["population"]["draw"]["values"] = rows
    elif bad == "graph":
        raw["change"]["base"]["task"]["context_rows"] = 7
    elif bad == "extra":
        raw["extra"] = None
    elif bad == "geometry":
        raw["change"]["interventions"][0]["channels"] = (999,)
    else:
        limits = p.CausalLimits(parameters=1, working_bytes=1)

    def forbidden(*args: object, **kwargs: object):
        pytest.fail("known-invalid GP comparison performed random or covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np.random, "Generator", forbidden)
    with pytest.raises(
        ValueError, match="GP|intervention|original|continuous|limits|unsupported recipe version"
    ):
        p.recipe_from_dict(raw, limits=limits)


def test_raw_gp_intervention_constructor_preserves_archive(archive: dict[str, Any]):
    result = p.construct_gp_root_intervention(
        {"source": archive["base"], "change": archive["change"]},
        seed=47,
        task_id=59,
        limits=p.CausalLimits(),
    )
    assert asdict(result) == archive


@pytest.mark.parametrize("identity", [(True, 59), (47, 59.0), (47, 60)])
def test_raw_gp_intervention_identity_precedes_replay(
    archive: dict[str, Any],
    identity: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    def forbidden(*args: object, **kwargs: object):
        pytest.fail("known bad identity reached covariance work")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    with pytest.raises(ValueError, match="identity"):
        p.construct_gp_root_intervention(
            {"source": archive["base"], "change": archive["change"]},
            seed=identity[0],
            task_id=identity[1],
            limits=p.CausalLimits(),
        )
