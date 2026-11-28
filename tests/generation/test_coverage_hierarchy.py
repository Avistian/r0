"""Hierarchical budgets differ from counting every configured leaf equally."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from r0.modules import generation as g
from r0.modules import priors as p


def campaign() -> g.CoveragePlan:
    request = g.RunRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=18,
        count=1,
        max_seconds=30,
        max_bytes=8000000,
        sampling=g.RecipeSampling("polynomial"),
    )
    strata = tuple(
        g.CoverageStratum(key, request, minimum=0, tags=(("family", family), ("regime", regime)))
        for key, family, regime in (
            ("a-low", "a", "low"),
            ("a-high", "a", "high"),
            ("b-low", "b", "low"),
        )
    )
    return g.CoveragePlan(strata, total=12)


def counts(plan: g.CoveragePlan) -> dict[str, int]:
    return {a.key: a.count for a in g.coverage_allocations(plan)}


def test_conditional_budgets_do_not_reward_more_variants() -> None:
    flat = campaign()
    assert counts(flat) == {"a-high": 4, "a-low": 4, "b-low": 4}
    hierarchy = g.CoverageHierarchy(("family", "regime"))
    nested = replace(flat, hierarchy=hierarchy, version="coverage-plan-v2")
    assert counts(nested) == {"a-high": 3, "a-low": 3, "b-low": 6}
    weighted = replace(nested, hierarchy=replace(hierarchy, weights=((("a",), 3), (("b",), 1))))
    assert counts(weighted) == {"a-high": 5, "a-low": 4, "b-low": 3}
    conditional = replace(nested, hierarchy=replace(hierarchy, weights=((("a", "low"), 2),)))
    assert counts(conditional) == {"a-high": 2, "a-low": 4, "b-low": 6}


def test_floors_survive_zero_branch_weight_and_json_roundtrip() -> None:
    base = campaign()
    strata = tuple(replace(s, minimum=2 if s.key == "b-low" else 0) for s in base.strata)
    hierarchy = g.CoverageHierarchy(("family",), weights=((("b",), 0), (("a",), 1)))
    plan = replace(base, strata=strata, hierarchy=hierarchy, version="coverage-plan-v2")
    assert counts(plan) == {"a-high": 5, "a-low": 5, "b-low": 2}
    assert g.coverage_plan_from_json(plan.to_json()).identity == plan.identity
    reordered = replace(
        plan,
        strata=tuple(reversed(strata)),
        hierarchy=replace(hierarchy, weights=tuple(reversed(hierarchy.weights))),
    )
    assert reordered.identity == plan.identity
    assert '"hierarchy"' not in base.to_json()
    assert g.coverage_plan_from_json(base.to_json()).to_json() == base.to_json()
    large = replace(plan, total=100_000_000)
    assert sum(counts(large).values()) == 100_000_000
    assert counts(large)["b-low"] == 2


def test_bad_branches_dimensions_versions_and_zero_residual_weights() -> None:
    base = campaign()
    for hierarchy in (
        g.CoverageHierarchy(("unknown",)),
        g.CoverageHierarchy(("family",), weights=((("missing",), 1),)),
        g.CoverageHierarchy(("family",), weights=((("a",), 0), (("b",), 0))),
    ):
        with pytest.raises(ValueError, match="dimension|nonexistent|no weight"):
            replace(base, hierarchy=hierarchy, version="coverage-plan-v2")
    with pytest.raises(ValueError, match="invalid bounded"):
        replace(base, hierarchy=g.CoverageHierarchy(("family",)))
    with pytest.raises(ValueError, match="invalid bounded"):
        replace(base, version="coverage-plan-v2")
    with pytest.raises(ValueError, match="duplicate"):
        g.CoverageHierarchy(("family",), weights=((("a",), 1), (("a",), 2)))
    zero_leaf = tuple(replace(s, weight=0) if s.key.startswith("a") else s for s in base.strata)
    with pytest.raises(ValueError, match="no weight"):
        replace(
            base,
            strata=zero_leaf,
            version="coverage-plan-v2",
            hierarchy=g.CoverageHierarchy(("family",)),
        )
    data = json.loads(base.to_json())
    data["hierarchy"] = {"dimensions": ["family"], "weights": []}
    with pytest.raises(ValueError, match="fields"):
        g.coverage_plan_from_json(json.dumps(data))


def test_hierarchical_plan_uses_same_guarded_persistence(tmp_path: Path) -> None:
    plan = replace(
        campaign(),
        total=3,
        shard_size=1,
        version="coverage-plan-v2",
        hierarchy=g.CoverageHierarchy(("family",)),
    )
    first = g.run_coverage_shard(plan, 0, tmp_path)
    assert first.returncode == 0, first.stderr
    again = g.run_coverage_shard(plan, 0, tmp_path)
    assert again.returncode == 0, again.stderr
    assert again.summary is not None
    assert again.summary["generated"] == 0
    report = g.inspect_coverage(plan, tmp_path)
    assert report["totals"]["committed"] == 1
    assert report["totals"]["pending"] == 2
