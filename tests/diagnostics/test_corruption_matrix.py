"""Exercise diagnostic sensitivity across families, task types and noise regimes."""

import json
from dataclasses import replace
from typing import Any

import pytest

from r0.modules.diagnostics import Outcome, diagnose
from r0.modules.episodes import Target, TaskType
from r0.modules.priors import FAMILIES, GeneratedTask, Proposal, sample_proposal

TASK_TYPES: tuple[TaskType, ...] = ("regression", "binary", "multiclass")
PROPOSALS = tuple(
    Proposal(family, task_type=task_type, noise=noise)
    for family in FAMILIES
    for task_type in TASK_TYPES
    for noise in ("none", "added")
)


def _corrupt_target(task: GeneratedTask) -> GeneratedTask:
    row_id = task.source_rows[0].id
    value: Target = (
        float(task.source_targets[0]) + 100.0
        if task.proposal.task_type == "regression"
        else next(c for c in task.episode.classes if c != task.source_targets[0])
    )
    episode = task.episode
    return replace(
        task,
        source_targets=(value,) + task.source_targets[1:],
        episode=replace(
            episode,
            context_targets=tuple(
                value if row.id == row_id else y
                for row, y in zip(episode.context, episode.context_targets, strict=True)
            ),
            query_targets=tuple(
                value if row.id == row_id else y
                for row, y in zip(episode.queries, episode.query_targets, strict=True)
            ),
        ),
    )


@pytest.mark.parametrize("proposal", PROPOSALS, ids=lambda p: f"{p.family}-{p.task_type}-{p.noise}")
def test_record_corruptions_fail_and_legitimate_control_passes(proposal: Proposal) -> None:
    task = sample_proposal(proposal)
    control = diagnose((Outcome(proposal, task=task),), (proposal,))
    assert control["status"] == "PASS"
    assert control["totals"]["valid"] == 1
    episode = task.episode
    wrong_partition = replace(
        episode,
        context=episode.context[:-1],
        context_targets=episode.context_targets[:-1],
        queries=episode.context[-1:] + episode.queries,
        query_targets=episode.context_targets[-1:] + episode.query_targets,
    )
    trace: dict[str, Any] = json.loads(task.trace_json)
    trace["latent"][0] += 100.0
    cases = (
        (replace(task, streams=()), "STREAM_INTEGRITY"),
        (replace(task, episode=wrong_partition), "PARTITION_INTEGRITY"),
        (_corrupt_target(task), "LABEL_INTEGRITY"),
        (replace(task, trace_json=json.dumps(trace)), "LATENT_INTEGRITY"),
    )
    for corrupted, expected_code in cases:
        report = diagnose((Outcome(proposal, task=corrupted),), (proposal,))
        assert report["status"] == "FAIL", expected_code
        assert report["totals"]["invalid"] == 1
        assert report["totals"]["selected"] == 0
        assert expected_code in {f["code"] for f in report["findings"]}


@pytest.mark.parametrize("proposal", [p for p in PROPOSALS if p.family == "tree"])
def test_discrete_tree_trace_rejects_wrong_leaf_in_every_target_regime(proposal: Proposal) -> None:
    task = sample_proposal(proposal)
    trace = json.loads(task.trace_json)
    trace["leaf_indices"][0] = 999
    report = diagnose(
        (Outcome(proposal, task=replace(task, trace_json=json.dumps(trace))),), (proposal,)
    )
    assert report["status"] == "FAIL"
    assert "RECORD_INVALID" in {f["code"] for f in report["findings"]}
