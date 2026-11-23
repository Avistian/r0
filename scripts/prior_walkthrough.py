"""Bounded human-review evidence for generator construction and diagnostic sensitivity."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from prior_calculations import worked_row

from r0.modules.diagnostics import Outcome, diagnose, inspect_task, task_identities
from r0.modules.priors import FAMILIES, GeneratedTask, Proposal, sample_proposal
from r0.modules.replay import diagnostic_schedule, parse_diagnostic_config, write_prior_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reference = write_prior_diagnostics(
        ROOT / "configs/diagnostics/coverage.toml", args.out / "reference"
    )
    biased = write_prior_diagnostics(
        ROOT / "configs/diagnostics/biased-filter.toml", args.out / "biased"
    )
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    check(
        "default manifest",
        reference["status"] == "PASS"
        and reference["totals"]["selected"] == 72
        and all(c["counts"]["proposed"] == 2 for c in reference["coverage"]),
        reference["totals"],
    )
    check(
        "biased filtering",
        biased["totals"]["valid"] == 72
        and biased["totals"]["selected"] == 36
        and sum(f["code"] == "SELECTION_REMOVED_CELL" for f in biased["findings"]) == 18,
        {
            "totals": biased["totals"],
            "affected_cells": [c for c in biased["coverage"] if "/added/" in c["cell"]],
        },
    )
    proposals = diagnostic_schedule(
        parse_diagnostic_config((ROOT / "configs/diagnostics/coverage.toml").read_text())
    )
    tasks = {p.id: sample_proposal(p) for p in reversed(proposals)}
    replay_ok = True
    for p in proposals:
        card = json.loads((args.out / "reference/proposals" / f"{p.id}.json").read_text())
        task = tasks[p.id]
        replayed = {
            "source_rows": [asdict(r) for r in task.source_rows],
            "source_targets": task.source_targets,
            "episode": asdict(task.episode),
            "predictor_view": asdict(task.episode.predictor_view()),
            "mechanism": json.loads(task.mechanism_json),
            "trace": json.loads(task.trace_json),
            "streams": task.streams,
        }
        replay_ok = replay_ok and json.loads(json.dumps(replayed)) == {
            key: card[key] for key in replayed
        }
    check("request-order replay", replay_ok, {"compared_proposals": len(tasks)})
    check(
        "same proposals under selection",
        [r["identities"] for r in reference["records"]]
        == [r["identities"] for r in biased["records"]],
        "all content/mechanism/equivalence identities agree",
    )
    pair_checks = []
    for p in proposals[::2]:
        clean, noisy = tasks[p.id], tasks[replace(p, noise="added").id]
        pair_checks.append(
            clean.mechanism_id == noisy.mechanism_id
            and clean.source_rows == noisy.source_rows
            and clean.episode.context == noisy.episode.context
        )
    check(
        "paired noise and legitimate noisy tasks",
        all(pair_checks),
        {"pairs": len(pair_checks), "invalid": reference["totals"]["invalid"]},
    )
    t = tasks[proposals[0].id]
    duplicate = sample_proposal(replace(t.proposal, split=1))
    duplicate_report = diagnose(
        (Outcome(t.proposal, task=t), Outcome(duplicate.proposal, task=duplicate)),
        (t.proposal, duplicate.proposal),
    )
    save(args.out / "duplicate.json", duplicate_report)
    check(
        "exact content with new partition",
        duplicate_report["duplicates"]["content"]["excess_instances"] == 1,
        {
            "findings": duplicate_report["findings"],
            "episode_ids": [t.episode.episode_id, duplicate.episode.episode_id],
        },
    )
    new_observations = sample_proposal(replace(t.proposal, observation=1))
    old_ids, new_ids = task_identities(t), task_identities(new_observations)
    check(
        "same mechanism new observations",
        old_ids["mechanism"] == new_ids["mechanism"] and old_ids["content"] != new_ids["content"],
        {"original": old_ids, "new": new_ids},
    )
    reordered = replace(t, source_rows=t.source_rows[::-1], source_targets=t.source_targets[::-1])
    check(
        "row permutation equivalence",
        task_identities(reordered)["equivalence"] == old_ids["equivalence"]
        and task_identities(reordered)["content"] != old_ids["content"],
        task_identities(reordered),
    )
    binary = next(t for t in tasks.values() if t.proposal.task_type == "binary")
    names = dict(zip(binary.episode.classes, ("renamed-b", "renamed-a"), strict=True))
    renamed = replace(
        binary,
        source_targets=tuple(names[str(y)] for y in binary.source_targets),
        episode=replace(
            binary.episode,
            classes=tuple(names[c] for c in binary.episode.classes),
            context_targets=tuple(names[str(y)] for y in binary.episode.context_targets),
            query_targets=tuple(names[str(y)] for y in binary.episode.query_targets),
        ),
    )
    check(
        "bijective class renaming",
        task_identities(renamed)["equivalence"] == task_identities(binary)["equivalence"]
        and not inspect_task(renamed)[0],
        {"mapping": names, "identities": task_identities(renamed)},
    )
    changed_id = t.source_rows[0].id
    changed_value = float(t.source_targets[0]) + 10
    corrupted = replace(
        t,
        source_targets=(changed_value,) + t.source_targets[1:],
        episode=replace(
            t.episode,
            context_targets=tuple(
                changed_value if r.id == changed_id else y
                for r, y in zip(t.episode.context, t.episode.context_targets, strict=True)
            ),
            query_targets=tuple(
                changed_value if r.id == changed_id else y
                for r, y in zip(t.episode.queries, t.episode.query_targets, strict=True)
            ),
        ),
    )
    corrupt_report = diagnose((Outcome(t.proposal, task=corrupted),), (t.proposal,))
    save(
        args.out / "corrupt-label.json",
        {"report": corrupt_report, "original": asdict(t), "corrupted": asdict(corrupted)},
    )
    check(
        "corrupted target",
        corrupt_report["status"] == "FAIL"
        and any(f["code"] == "LABEL_INTEGRITY" for f in corrupt_report["findings"]),
        corrupt_report["findings"],
    )
    missing = diagnose((), (t.proposal,))
    save(args.out / "missing-proposal.json", missing)
    check(
        "lost proposal",
        missing["status"] == "FAIL" and missing["totals"]["generated"] == 0,
        missing["findings"],
    )
    isolated = replace(
        t.episode, query_targets=tuple(float(y) + 10 for y in t.episode.query_targets)
    )
    check(
        "query-target isolation",
        isolated.predictor_view() == t.episode.predictor_view(),
        "query targets changed; predictor view exactly unchanged",
    )
    cards = []
    for family in FAMILIES:
        task = next(
            t
            for t in tasks.values()
            if t.proposal.family == family
            and t.proposal.noise == "added"
            and t.proposal.task_type == "regression"
        )
        card = json.loads(
            (args.out / "reference/proposals" / f"{task.proposal.id}.json").read_text()
        )
        cards.append(worked_row(card))
        # A second card exercises discrete threshold/mapping/replacement semantics.
        classified = next(
            t
            for t in tasks.values()
            if t.proposal.family == family
            and t.proposal.task_type == "multiclass"
            and t.proposal.noise == "added"
        )
        cards.append(
            worked_row(
                json.loads(
                    (
                        args.out / "reference/proposals" / f"{classified.proposal.id}.json"
                    ).read_text()
                )
            )
        )
    save(args.out / "worked-calculations.json", cards)
    check(
        "independent worked rows",
        all(c["status"] == "PASS" for c in cards),
        [
            {"family": c["family"], "proposal_id": c["proposal_id"], "status": c["status"]}
            for c in cards
        ],
    )

    def broken(p: Proposal) -> GeneratedTask:
        raise ValueError(f"deliberate generation failure for {p.id}")

    failure_config = args.out / "failure.toml"
    failure_config.write_text('version=1\nfamilies=["linear"]\nseeds=[11]\ndraws=1\n')
    failed = write_prior_diagnostics(
        failure_config, args.out / "generation-failure", sampler=broken
    )
    check(
        "generation failure retention",
        failed["status"] == "FAIL" and failed["totals"]["generation_failed"] == 2,
        failed["totals"],
    )
    variants = {
        family: sorted(
            {
                json.loads(t.mechanism_json)["parameters"][key]
                for t in tasks.values()
                if t.proposal.family == family
            }
        )
        for family, key in (("scm", "motif"), ("interaction", "rule"))
    }
    supplemental_config = args.out / "supplemental-fork.toml"
    supplemental_config.write_text('version=1\nfamilies=["scm"]\nseeds=[0]\ndraws=1\n')
    supplemental = write_prior_diagnostics(supplemental_config, args.out / "supplemental-fork")
    fork = sample_proposal(Proposal("scm", seed=0))
    check(
        "representative variants",
        variants["interaction"] == ["gated", "parity", "product"]
        and json.loads(fork.mechanism_json)["parameters"]["motif"] == "fork"
        and supplemental["status"] == "PASS",
        {
            "default_observed": variants,
            "default_missing": {"scm": ["fork"]},
            "supplemental_fork_proposal": fork.proposal.id,
            "scope": "separate declared example, not a replacement or top-up of the 72 proposals",
        },
    )
    summary = {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "checks": checks,
        "reference_run_id": reference["run_id"],
        "biased_run_id": biased["run_id"],
        "observed_elapsed_seconds": time.perf_counter() - started,
        "downstream_usefulness": "NOT_MEASURED",
        "superiority": "NOT_MEASURED",
    }
    save(args.out / "walkthrough.json", summary)
    print(json.dumps({"status": summary["status"], "checks": len(checks), "out": str(args.out)}))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
