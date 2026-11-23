from dataclasses import replace

from r0.modules.diagnostics import Outcome, diagnose
from r0.modules.priors import Proposal, sample_proposal


def test_diagnostics_expose_selection_loss_and_corrupt_labels() -> None:
    proposals = (Proposal("linear"), Proposal("linear", noise="added"))
    tasks = tuple(sample_proposal(p) for p in proposals)
    outcomes = tuple(Outcome(p, task=t) for p, t in zip(proposals, tasks, strict=True))
    report = diagnose(outcomes, proposals, selection="drop_added_noise")
    assert report["totals"] == {
        "proposed": 2,
        "generated": 2,
        "valid": 2,
        "selected": 1,
        "generation_failed": 0,
        "invalid": 0,
        "selection_rejected": 1,
    }
    assert any(f["code"] == "SELECTION_REMOVED_CELL" for f in report["findings"])
    corrupted = replace(tasks[0], source_targets=(12345.0,) + tasks[0].source_targets[1:])
    bad = diagnose((Outcome(proposals[0], task=corrupted),), proposals[:1])
    assert bad["status"] == "FAIL"
    assert any(f["code"] == "LABEL_INTEGRITY" for f in bad["findings"])
    lost = diagnose(outcomes[:1], proposals)
    assert lost["status"] == "FAIL"
    assert any(f["code"] == "MISSING_PROPOSAL" for f in lost["findings"])


def test_duplicate_definitions_distinguish_rows_mechanisms_and_label_names() -> None:
    from r0.modules.diagnostics import task_identities

    task = sample_proposal(Proposal("linear", task_type="binary"))
    reordered = replace(
        task, source_rows=task.source_rows[::-1], source_targets=task.source_targets[::-1]
    )
    original = task_identities(task)
    assert task_identities(reordered)["content"] != original["content"]
    assert task_identities(reordered)["equivalence"] == original["equivalence"]
    episode = task.episode
    rename = dict(zip(episode.classes, ("renamed-b", "renamed-a"), strict=True))
    renamed = replace(
        task,
        source_targets=tuple(rename[str(y)] for y in task.source_targets),
        episode=replace(
            episode,
            classes=tuple(rename[c] for c in episode.classes),
            context_targets=tuple(rename[str(y)] for y in episode.context_targets),
            query_targets=tuple(rename[str(y)] for y in episode.query_targets),
        ),
    )
    assert task_identities(renamed)["equivalence"] == original["equivalence"]
    new_data = sample_proposal(replace(task.proposal, observation=1))
    assert task_identities(new_data)["mechanism"] == original["mechanism"]
    assert task_identities(new_data)["content"] != original["content"]


def test_valid_noisy_representatives_and_failed_proposal_accounting() -> None:
    from r0.modules.priors import FAMILIES

    proposals = tuple(Proposal(f, noise="added") for f in FAMILIES)
    outcomes = tuple(Outcome(p, task=sample_proposal(p)) for p in proposals)
    assert diagnose(outcomes, proposals)["status"] == "PASS"
    failed = (Outcome(proposals[0], generation_error="synthetic failure injection"),) + outcomes[1:]
    report = diagnose(failed, proposals)
    assert report["totals"]["generation_failed"] == 1
    assert report["totals"]["selection_rejected"] == 0
    assert report["status"] == "FAIL"


def test_rejects_changed_partition_stream_metadata_and_tree_path() -> None:
    import json

    p = Proposal("tree")
    task = sample_proposal(p)
    episode = task.episode
    changed = replace(
        episode,
        context=episode.context[:-1],
        context_targets=episode.context_targets[:-1],
        queries=episode.context[-1:] + episode.queries,
        query_targets=episode.context_targets[-1:] + episode.query_targets,
    )
    trace = json.loads(task.trace_json)
    trace["leaf_indices"] = [999] * p.rows
    corruptions = (
        replace(task, episode=changed),
        replace(task, streams=()),
        replace(task, trace_json=json.dumps(trace)),
        replace(
            task,
            episode=replace(
                episode,
                context=episode.context[::-1],
                context_targets=episode.context_targets[::-1],
            ),
        ),
    )
    for corrupted in corruptions:
        report = diagnose((Outcome(p, task=corrupted),), (p,))
        assert report["status"] == "FAIL"
        assert report["totals"]["invalid"] == 1
