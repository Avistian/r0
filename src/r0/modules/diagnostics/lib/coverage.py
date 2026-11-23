"""Proposal accounting and explicitly limited duplicate/equivalence diagnostics."""

import itertools
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from r0.modules.diagnostics.lib.verification import Finding, inspect_task
from r0.modules.priors import GeneratedTask, Proposal, record_fingerprint


@dataclass(frozen=True, slots=True)
class Outcome:
    proposal: Proposal
    task: GeneratedTask | None = None
    generation_error: str | None = None

    def __post_init__(self) -> None:
        if (self.task is None) == (self.generation_error is None):
            raise ValueError("outcome requires exactly one generated task or generation error")
        if self.generation_error is not None and not self.generation_error.strip():
            raise ValueError("generation failure requires an explanation")


def task_identities(task: GeneratedTask) -> dict[str, str]:
    schema = [asdict(f) for f in task.episode.features]
    content = [(r.values, y) for r, y in zip(task.source_rows, task.source_targets, strict=True)]
    exact = record_fingerprint((schema, task.episode.task_type, task.episode.classes, content))
    # Bounded vocabulary (<=3), so enumerate bijections without heuristic matching.
    classes = task.episode.classes
    if len(classes) > 3:
        raise ValueError("equivalence diagnostic supports at most three classes")
    labelings = itertools.permutations(range(len(classes))) if classes else [()]
    equivalents: list[str] = []
    for labels in labelings:
        mapping = dict(zip(classes, labels, strict=True))
        rows = sorted(
            record_fingerprint((r.values, mapping[str(y)] if classes else y))
            for r, y in zip(task.source_rows, task.source_targets, strict=True)
        )
        equivalents.append(record_fingerprint((schema, task.episode.task_type, len(classes), rows)))
    return {"content": exact, "mechanism": task.mechanism_id, "equivalence": min(equivalents)}


def _cell(p: Proposal) -> str:
    return f"{p.family}/{p.noise}/{p.task_type}"


def _share(count: int, denominator: int) -> dict[str, object]:
    return {
        "value": count / denominator if denominator else None,
        "count": count,
        "denominator": denominator,
        "reason": None if denominator else "zero denominator",
    }


def diagnose(
    outcomes: tuple[Outcome, ...],
    expected: tuple[Proposal, ...],
    *,
    selection: Literal["none", "drop_added_noise"] = "none",
) -> dict[str, Any]:
    if selection not in ("none", "drop_added_noise"):
        raise ValueError("unsupported diagnostic selection policy")
    if not 1 <= len(expected) <= 256 or len(outcomes) > 256:
        raise ValueError(
            "diagnostic run limit is 1..256 expected proposals and at most 256 outcomes"
        )
    expected_by_id = {p.id: p for p in expected}
    if len(expected_by_id) != len(expected):
        raise ValueError("expectation manifest contains duplicate proposal identities")
    findings: list[Finding] = []
    counts = {
        "proposed": len(expected),
        "generated": 0,
        "valid": 0,
        "selected": 0,
        "generation_failed": 0,
        "invalid": 0,
        "selection_rejected": 0,
    }
    cells: dict[str, dict[str, int]] = {}
    for p in expected:
        cell = cells.setdefault(_cell(p), dict.fromkeys(counts, 0))
        cell["proposed"] += 1
    observed: set[str] = set()
    records: list[dict[str, Any]] = []
    fingerprints: dict[str, Counter[str]] = {
        name: Counter() for name in ("content", "mechanism", "equivalence")
    }
    for outcome in outcomes:
        p = outcome.proposal
        if p.id not in expected_by_id or p.id in observed:
            findings.append(
                Finding(
                    "PROPOSAL_ACCOUNTING", "error", "unexpected or repeated proposal identity", p.id
                )
            )
            continue
        observed.add(p.id)
        cell = cells[_cell(p)]
        record: dict[str, Any] = {
            "proposal_id": p.id,
            "proposal": asdict(p),
            "cell": _cell(p),
            "generation_error": outcome.generation_error,
            "selected": False,
            "selection_reason": None,
            "descriptors": {},
            "findings": [],
        }
        if outcome.task is None:
            counts["generation_failed"] += 1
            cell["generation_failed"] += 1
            findings.append(
                Finding("GENERATION_FAILED", "error", str(outcome.generation_error), p.id)
            )
            record["generation_status"] = "failed"
            records.append(record)
            continue
        record["generation_status"] = "generated"
        counts["generated"] += 1
        cell["generated"] += 1
        local, descriptors = inspect_task(outcome.task)
        if outcome.task.proposal != p:
            local += (
                Finding("RECORD_INVALID", "error", "task proposal differs from outcome", p.id),
            )
        findings.extend(local)
        record["descriptors"] = descriptors
        record["findings"] = [asdict(f) for f in local]
        record["valid"] = not local
        if local:
            counts["invalid"] += 1
            cell["invalid"] += 1
        else:
            counts["valid"] += 1
            cell["valid"] += 1
            identities = task_identities(outcome.task)
            record["identities"] = identities
            for name, identity in identities.items():
                fingerprints[name][identity] += 1
            selected = selection == "none" or p.noise != "added"
            record["selected"] = selected
            key = "selected" if selected else "selection_rejected"
            counts[key] += 1
            cell[key] += 1
            record["selection_reason"] = (
                "retained" if selected else "demonstration filter: added observation noise"
            )
        records.append(record)
    for identity in sorted(expected_by_id.keys() - observed):
        findings.append(
            Finding("MISSING_PROPOSAL", "error", "expected proposal has no outcome", identity)
        )
    coverage: list[dict[str, Any]] = []
    for name, cell in sorted(cells.items()):
        if cell["valid"] and not cell["selected"]:
            findings.append(Finding("SELECTION_REMOVED_CELL", "warning", name))
        coverage.append(
            {
                "cell": name,
                "state": "observed" if cell["generated"] else "zero_observed",
                "counts": cell,
                "shares": {
                    stage: _share(cell[stage], counts[stage])
                    for stage in ("proposed", "generated", "valid", "selected")
                },
                "within_cell_retention": _share(cell["selected"], cell["valid"]),
            }
        )
    duplicates: dict[str, object] = {}
    for name, counter in fingerprints.items():
        repeated = {key: value for key, value in sorted(counter.items()) if value > 1}
        duplicates[name] = {
            "unique": len(counter),
            "repeated": repeated,
            "excess_instances": sum(value - 1 for value in counter.values()),
        }
        if repeated:
            findings.append(
                Finding(
                    f"REPEATED_{name.upper()}",
                    "warning",
                    f"{len(repeated)} repeated {name} groups; not automatic invalidity",
                )
            )
    return {
        "status": "FAIL" if any(f.severity == "error" for f in findings) else "PASS",
        "totals": counts,
        "selection": selection,
        "coverage": coverage,
        "records": records,
        "duplicates": duplicates,
        "findings": [asdict(f) for f in findings],
        "measured_axes": ["family", "added_observation_noise", "task_type"],
        "descriptor_axes": ["sparsity", "nuisance_features", "class_counts", "sample_variances"],
        "catalog_only_axes": [
            "missingness_process",
            "subpopulations",
            "context_query_shift",
            "relational",
        ],
        "downstream_usefulness": "NOT_MEASURED",
    }
