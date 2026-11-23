"""Immutable bounded run artifacts, preserving failed proposals before reporting failure."""

import hashlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from r0.modules.diagnostics import Outcome, diagnose
from r0.modules.priors import (
    GeneratedTask,
    Proposal,
    diagnostic_profile,
    record_fingerprint,
    sample_proposal,
)
from r0.modules.replay.lib.diagnostic_config import diagnostic_schedule, parse_diagnostic_config


def _write(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(encoded)


def _coverage_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Generator coverage",
        "",
        f"Status: {report['status']}. Selection: {report['selection']}.",
        "Downstream usefulness: NOT_MEASURED. Counts describe this bounded run only.",
        "",
        "| Cell | Proposed | Generated | Valid | Selected | Selected share |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for cell in report["coverage"]:
        counts, share = cell["counts"], cell["shares"]["selected"]
        lines.append(
            f"| {cell['cell']} | {counts['proposed']} | {counts['generated']} | "
            f"{counts['valid']} | {counts['selected']} | "
            f"{share['count']}/{share['denominator']} ({share['value']}) |"
        )
    lines += ["", "## Findings", ""]
    lines += [f"- {f['severity']}: {f['code']}: {f['detail']}" for f in report["findings"]]
    lines += [
        "",
        "Shares have explicit stage denominators in report.json; null means a zero denominator.",
        "Missingness processes, subgroup and shift coverage remain catalog-only, not measured.",
    ]
    return "\n".join(lines) + "\n"


def write_prior_diagnostics(
    config_path: Path,
    output_path: Path,
    *,
    sampler: Callable[[Proposal], GeneratedTask] = sample_proposal,
) -> dict[str, Any]:
    config = parse_diagnostic_config(config_path.read_text())
    proposals = diagnostic_schedule(config)
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError("existing run directory is immutable; choose a new output path") from None
    resolved = {
        **asdict(config),
        "resolved_profile": diagnostic_profile(),
        "expectation": [asdict(p) for p in proposals],
        "numeric_tolerance": {"atol": 1e-10, "rtol": 1e-10},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    _write(output_path / "config.json", resolved)
    run_id = record_fingerprint(resolved)
    directory = output_path / "proposals"
    directory.mkdir()
    outcomes: list[Outcome] = []
    started = time.perf_counter()
    for p in proposals:
        task_started = time.perf_counter()
        task: GeneratedTask | None = None
        try:
            task = sampler(p)
            outcome = Outcome(p, task=task)
            payload: dict[str, Any] = {
                "proposal_id": p.id,
                "proposal": asdict(p),
                "generation_status": "generated",
                "source_rows": [asdict(row) for row in task.source_rows],
                "source_targets": task.source_targets,
                "episode": asdict(task.episode),
                "predictor_view": asdict(task.episode.predictor_view()),
                "mechanism": json.loads(task.mechanism_json),
                "trace": json.loads(task.trace_json),
                "streams": task.streams,
            }
            # Validate serialization before considering this a retained generated record.
            record_fingerprint(payload)
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            outcome = Outcome(p, generation_error=detail)
            payload = {
                "proposal_id": p.id,
                "proposal": asdict(p),
                "generation_status": "failed",
                "generation_error": detail,
                "returned_record_repr": repr(task) if task is not None else None,
            }
        outcomes.append(outcome)
        payload["semantic_fingerprint"] = record_fingerprint(payload)
        payload["observed_generation_seconds"] = time.perf_counter() - task_started
        _write(directory / f"{p.id}.json", payload)
    try:
        report = diagnose(tuple(outcomes), proposals, selection=config.selection)
    except Exception as error:
        _write(
            output_path / "fatal.json",
            {"status": "FAIL", "unexpected_error": f"{type(error).__name__}: {error}"},
        )
        raise ValueError(
            "unexpected diagnostic failure; proposal records and fatal.json retained"
        ) from error
    report["run_id"] = run_id
    report["config_fingerprint"] = record_fingerprint(resolved)
    report["observed_elapsed_seconds"] = time.perf_counter() - started
    report["cost_scope"] = (
        "observed serial CPU generation/diagnosis, not a portable performance threshold"
    )
    _write(output_path / "report.json", report)
    with (output_path / "coverage.md").open("x") as handle:
        handle.write(_coverage_markdown(report))
    files = sorted(p for p in output_path.rglob("*") if p.is_file())
    _write(
        output_path / "manifest.json",
        {
            "run_id": run_id,
            "files": {
                str(p.relative_to(output_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
        },
    )
    return report
