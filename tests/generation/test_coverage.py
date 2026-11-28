"""Exact quota arithmetic, global shard identities and actual guarded persistence."""

from dataclasses import replace
from pathlib import Path

import pytest

from r0.modules import generation as g
from r0.modules import priors as p


def template(seed: int = 30) -> g.RunRequest:
    return g.RunRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=seed,
        count=1,
        max_seconds=30,
        max_bytes=8000000,
        group_size=2,
        sampling=g.RecipeSampling("polynomial"),
    )


def strata() -> tuple[g.CoverageStratum, ...]:
    return (
        g.CoverageStratum("a", template(), weight=3, minimum=2, tags=(("family", "polynomial"),)),
        g.CoverageStratum("b", template(31), weight=1, minimum=1, tags=(("family", "polynomial"),)),
        g.CoverageStratum("rare", template(32), weight=0, minimum=2, tags=(("family", "rare"),)),
    )


def test_exact_quota_floors_ties_and_order() -> None:
    plan = g.CoveragePlan(strata(), total=12, shard_size=3, start=101)
    # Seven remaining tasks: (21/4, 7/4, 0), so floor counts plus the final
    # largest remainder yield 7,3,2 including explicit minimums.
    allocations = g.coverage_allocations(plan)
    assert [a.count for a in allocations] == [7, 3, 2]
    assert [a.start for a in allocations] == [101, 108, 111]
    assert g.coverage_allocations(replace(plan, strata=tuple(reversed(strata())))) == allocations
    assert replace(plan, strata=tuple(reversed(strata()))).identity == plan.identity
    tied = g.CoveragePlan(tuple(replace(s, weight=1, minimum=0) for s in strata()), total=2)
    assert [a.count for a in g.coverage_allocations(tied)] == [1, 1, 0]
    assert g.coverage_plan_from_json(plan.to_json()) == plan


def test_shards_partition_ids_once_and_keep_templates() -> None:
    plan = g.CoveragePlan(strata(), total=12, shard_size=3, start=101)
    report = g.planned_coverage(plan)
    assert report["status"] == "PLANNED_NOT_GENERATED"
    assert report["marginals"] == {"family": {"polynomial": 10, "rare": 2}}
    ids: list[int] = []
    for index in range(report["shards"]):
        shard = g.coverage_shard(plan, index)
        assert shard.index == index
        assert shard.plan_id == plan.identity
        assert shard.request.count <= 3
        ids.extend(range(shard.request.start, shard.request.start + shard.request.count))
    assert ids == list(range(101, 113))
    assert all(s.request.count == 1 and s.request.start == 0 for s in plan.strata)
    with pytest.raises(ValueError, match="outside"):
        g.coverage_shard(plan, report["shards"])
    with pytest.raises(ValueError, match="index"):
        g.coverage_shard(plan, True)


def test_100m_plan_is_small_and_random_access() -> None:
    plan = g.CoveragePlan(strata(), total=100_000_000, shard_size=10000)
    allocations = g.coverage_allocations(plan)
    assert sum(a.count for a in allocations) == 100_000_000
    assert allocations[-1].count == 2
    assert len(plan.to_json()) < 10000
    last = g.coverage_shard(plan, sum(a.shards for a in allocations) - 1)
    assert last.request.start + last.request.count == 100_000_000
    assert last.stratum == "rare"


def test_invalid_plans_and_independence_claims() -> None:
    with pytest.raises(ValueError, match="minimums"):
        g.CoveragePlan(strata(), total=4)
    with pytest.raises(ValueError, match="weight"):
        g.CoveragePlan(tuple(replace(s, weight=0) for s in strata()), total=6)
    with pytest.raises(ValueError, match="dimensions"):
        g.CoveragePlan((strata()[0], replace(strata()[1], tags=())), total=6)
    frozen = replace(template(), sampling=None)
    with pytest.raises(ValueError, match="independent-source"):
        g.CoveragePlan((g.CoverageStratum("fixed", frozen),), total=2)
    explicit = g.CoveragePlan((g.CoverageStratum("fixed", frozen),), total=2, require_fresh=False)
    assert not g.planned_coverage(explicit)["independent_source_sampling_required"]
    with pytest.raises(ValueError, match="fields"):
        g.coverage_plan_from_json(plan_with_extra_field(explicit))


def plan_with_extra_field(plan: g.CoveragePlan) -> str:
    return plan.to_json()[:-1] + ',"ignored":1}'


def test_guarded_shards_resume_and_reject_changed_plan(tmp_path: Path) -> None:
    plan = g.CoveragePlan(
        (g.CoverageStratum("sampled", template()),), total=3, shard_size=2, start=500
    )
    first = g.run_coverage_shard(plan, 0, tmp_path)
    assert first.returncode == 0, first.stderr
    assert first.summary is not None
    assert first.summary["completed"] == 2
    resumed = g.run_coverage_shard(plan, 0, tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.summary is not None
    assert resumed.summary["generated"] == 0
    second = g.run_coverage_shard(plan, 1, tmp_path)
    assert second.returncode == 0, second.stderr
    ids: list[int] = []
    recipes: list[str] = []
    for path in sorted((tmp_path / "shards").rglob("data-*.zip")):
        ids.extend(g.verify_group(path).task_ids)
        recipes.extend(task.recipe_json for task in g.load_group(path))
    assert ids == [500, 501, 502]
    assert len(set(recipes)) == 3
    with pytest.raises(ValueError, match="different plan"):
        g.run_coverage_shard(replace(plan, total=4), 0, tmp_path)


def test_joint_coverage_exposes_missing_cross_combinations() -> None:
    tagged = (
        replace(strata()[0], tags=(("family", "polynomial"), ("noise", "low"))),
        replace(strata()[1], tags=(("family", "polynomial"), ("noise", "high"))),
        replace(strata()[2], tags=(("family", "rare"), ("noise", "high"))),
    )
    plan = g.CoveragePlan(tagged, total=12)
    report = g.planned_coverage(plan, interactions=(("noise", "family"),))
    assert report["interactions"] == [
        {
            "dimensions": ["family", "noise"],
            "cells": [
                {"values": ["polynomial", "high"], "count": 3},
                {"values": ["polynomial", "low"], "count": 7},
                {"values": ["rare", "high"], "count": 2},
            ],
        }
    ]
    # No rare/low cell is invented from independently nonzero marginals.
    with pytest.raises(ValueError, match="dimensions"):
        g.planned_coverage(plan, interactions=(("family", "unknown"),))
