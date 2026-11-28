"""Duplicate content is independent of task IDs and persistence metadata."""

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from r0.modules import generation as g
from r0.modules import priors as p


def template() -> g.RunRequest:
    return g.RunRequest(
        p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2),
        seed=88,
        count=1,
        max_seconds=30,
        max_bytes=8000000,
    )


def test_signature_excludes_identity_but_includes_targets_and_split() -> None:
    task = p.generate_task(template().recipe, seed=88, task_id=0)
    signature = g.task_content_signature(task)
    renamed = replace(
        task, source_task_id="another", episode_id="another", recipe_json="{}", mechanism_json="{}"
    )
    assert g.task_content_signature(renamed) == signature
    assert g.task_content_signature(replace(task, targets=task.targets + 1)) != signature
    assert (
        g.task_content_signature(
            replace(task, context_indices=task.query_indices, query_indices=task.context_indices)
        )
        != signature
    )
    values = task.inputs.clone()
    values[0, 0] = 0.0
    positive = replace(task, inputs=values)
    negative_values = values.clone()
    negative_values[0, 0] = -0.0
    assert g.task_content_signature(positive) == g.task_content_signature(
        replace(task, inputs=negative_values)
    )
    changed = task.inputs.clone()
    changed[0, 0] = torch.nextafter(changed[0, 0], torch.tensor(float("inf")))
    assert g.task_content_signature(replace(task, inputs=changed)) != signature


def test_disk_index_counts_duplicate_fixture_and_is_idempotent(tmp_path: Path) -> None:
    request = template()
    plan = g.CoveragePlan((g.CoverageStratum("fixed", request),), total=2, require_fresh=False)
    (tmp_path / "coverage-plan.json").write_text(plan.to_json())
    reference = p.generate_task(request.recipe, seed=request.seed, task_id=0)

    def duplicate(task_id: int) -> p.TrainingTask:
        task = p.generate_task(request.recipe, seed=request.seed, task_id=task_id)
        # Deliberately repeated observations with independently valid identity records.
        return replace(
            task,
            inputs=reference.inputs,
            targets=reference.targets,
            context_indices=reference.context_indices,
            query_indices=reference.query_indices,
        )

    result = g.run_generation(
        g.coverage_shard(plan, 0).request, tmp_path / "shards" / "00000000", produce=duplicate
    )
    assert result.completed == 2
    unbound = tmp_path / "unbound.sqlite"
    with sqlite3.connect(unbound) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(ValueError, match="unbound"):
        g.index_coverage_shard(plan, 0, tmp_path, unbound)
    database = tmp_path / "content.sqlite"
    report = g.index_coverage_shard(plan, 0, tmp_path, database)
    assert report["tasks"] == 2
    assert report["unique_observed_tasks"] == 1
    assert report["duplicate_excess_fraction"] == 0.5
    assert report["duplicate_groups"] == 1
    assert report["largest_duplicate_group"] == 2
    assert report["rows"] == 32
    assert report["query_targets"] == 16
    assert report["mechanism_equivalence"] == "NOT_MEASURED"
    assert g.index_coverage_shard(plan, 0, tmp_path, database) == report
    assert (
        g.index_coverage_shard(plan, 0, tmp_path, database, summarize=False)["indexed_tasks"] == 2
    )
    # A persistent index must not silently rewrite an existing ID's signature.
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE tasks SET signature='changed' WHERE task_id=1")
    with pytest.raises(ValueError, match="content changed"):
        g.index_coverage_shard(plan, 0, tmp_path, database)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2
        connection.execute("UPDATE metadata SET value='wrong-plan' WHERE key='plan_id'")
    with pytest.raises(ValueError, match="different plan"):
        g.index_coverage_shard(plan, 0, tmp_path, database)


def test_live_pilot_content_index_survives_reopen(tmp_path: Path) -> None:
    request = replace(template(), sampling=g.RecipeSampling("polynomial"))
    plan = g.CoveragePlan((g.CoverageStratum("sampled", request),), total=2, shard_size=1)
    database = tmp_path / "content.sqlite"
    report = None
    for index in range(2):
        result = g.run_coverage_shard(plan, index, tmp_path)
        assert result.returncode == 0, result.stderr
        report = g.index_coverage_shard(plan, index, tmp_path, database)
        assert report["tasks"] == index + 1
    assert report is not None
    assert report["rows"] == 32
    assert report["query_targets"] == 16
    assert report["duplicate_excess"] == 0
    manifest = json.loads((tmp_path / "coverage-plan.json").read_text())
    assert manifest["total"] == 2
