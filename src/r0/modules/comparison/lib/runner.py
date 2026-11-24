"""Immutable attempts, explicit partial comparisons and exact-identity cache reads."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .contracts import validate_output
from .execution import environment, launch
from .external import ExternalPredictor
from .predictors import Unsupported, recipe
from .scoring import score
from .storage import Document, digest, file_hash, read_json, seal, verify, write_json
from .summary import aggregate_datasets, comparison_gaps, summarize_draws
from .tabpfn import tabpfn_status


def cache_identity(
    prepared: Path,
    predictor: str | ExternalPredictor,
    draw: int,
    *,
    diagnostic: bool,
) -> Document:
    verify(prepared)
    manifest = read_json(prepared / "manifest.json")
    if draw not in (1, 2, 3):
        raise ValueError("draw must be 1, 2 or 3")
    return {
        "version": 1,
        "manifest": digest(manifest),
        "draw": draw,
        "input": manifest["draws"][draw - 1]["sha256"],
        "scoring": manifest["scoring_sha256"],
        "predictor": predictor.snapshot()
        if isinstance(predictor, ExternalPredictor)
        else recipe(predictor, manifest["task"]),
        "environment": environment(),
        "limits": manifest["limits"],
        "mode": "FIXTURE_DIAGNOSTIC_UNENFORCED" if diagnostic else "ENFORCED",
    }


def read_attempt(path: Path, expected_identity: Document) -> Document:
    verify(path)
    result = read_json(path / "result.json")
    if result["identity"] != expected_identity:
        raise ValueError("cache identity mismatch; fresh authorized attempt required")
    if result["status"] != "SUCCESS":
        raise ValueError("failed attempts are retained but cannot be reused as successes")
    if file_hash(path / "input.json") != expected_identity["input"] or (
        file_hash(path / "scoring.json") != expected_identity["scoring"]
    ):
        raise ValueError("cache input/scoring identity mismatch")
    view = read_json(path / "input.json")
    labels = read_json(path / "scoring.json")
    output = read_json(path / "output/prediction.json")
    validate_output(output, view, expected_identity["predictor"].get("identity"))
    prediction = np.load(path / "output/predictions.npy", allow_pickle=False)
    recomputed = score(
        view["task"],
        view["classes"],
        labels["query_ids"],
        labels["targets"],
        output["query_ids"],
        prediction,
    )
    if any(
        not np.isclose(value, result["scores"][key], atol=1e-12, rtol=1e-10)
        for key, value in recomputed.items()
    ):
        raise ValueError("saved score differs from float64 recomputation")
    return result


def attempt(
    prepared: Path,
    destination: Path,
    predictor: str | ExternalPredictor,
    draw: int,
    *,
    cgroup_root: Path | None = None,
    diagnostic: bool = False,
) -> Document:
    """Execute once; never overwrite, resample, retry or silently reuse a result.

    diagnostic=True is explicitly unenforced fixture development, not a protocol run.
    """
    identity = cache_identity(prepared, predictor, draw, diagnostic=diagnostic)
    manifest = read_json(prepared / "manifest.json")
    if manifest["evidence"] != "AUTHORED_FIXTURE":
        raise ValueError("real comparison admission unavailable")
    destination.mkdir(parents=True, exist_ok=False)
    result: Document = {
        "version": 1,
        "identity": identity,
        "predictor": predictor.name if isinstance(predictor, ExternalPredictor) else predictor,
        "draw": draw,
        "evidence": "FIXTURE_ONLY",
        "status": "FAILED",
        "scores": None,
        "retry": "NOT_ATTEMPTED",
        "measurement": "NOT_PROTOCOL_EVIDENCE" if diagnostic else "FIXTURE_ENFORCED",
    }
    start = time.monotonic()
    try:
        source = prepared / manifest["draws"][draw - 1]["input"]
        if file_hash(source) != identity["input"]:
            raise ValueError("input hash mismatch")
        shutil.copyfile(source, destination / "input.json")
        command = (
            predictor.command()
            if isinstance(predictor, ExternalPredictor)
            else [
                sys.executable,
                "-m",
                "r0.modules.comparison",
                "worker",
                "--predictor",
                predictor,
            ]
        )
        receipt = launch(
            command,
            destination / "input.json",
            destination / "output",
            diagnostic=diagnostic,
            cgroup_root=cgroup_root,
        )
        result["resources"] = receipt
        if receipt.get("returncode") == 3:
            raise Unsupported(receipt["stdout"].strip())
        if receipt["status"] != "COMPLETED":
            raise ValueError(f"predictor process {receipt['status']}")
        shutil.copyfile(prepared / "scoring.json", destination / "scoring.json")
        labels = read_json(destination / "scoring.json")
        output = read_json(destination / "output/prediction.json")
        validate_output(
            output, read_json(destination / "input.json"), identity["predictor"].get("identity")
        )
        prediction = np.load(destination / "output/predictions.npy", allow_pickle=False)
        result["scores"] = score(
            manifest["task"],
            manifest["classes"],
            labels["query_ids"],
            labels["targets"],
            output["query_ids"],
            prediction,
        )
        result["status"] = "SUCCESS"
    except Unsupported as error:
        result["status"] = "UNSUPPORTED"
        result["reason"] = str(error)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    result["seconds_including_artifacts"] = time.monotonic() - start
    write_json(destination / "result.json", result)
    seal(destination)
    return result


def compare(
    prepared: Path,
    destination: Path,
    *,
    diagnostic: bool = False,
    cgroup_root: Path | None = None,
    fixture_predictors: bool = False,
    external_predictors: list[ExternalPredictor] | None = None,
    saved_attempts: dict[str, list[Path]] | None = None,
) -> Document:
    verify(prepared)
    descriptors = {d.name: d for d in external_predictors or []}
    if len(descriptors) != len(external_predictors or []) or (fixture_predictors and descriptors):
        raise ValueError("duplicate or conflicting predictor roles")
    if set(saved_attempts or {}) - set(descriptors):
        raise ValueError("saved attempts require expected predictor descriptors")
    destination.mkdir(parents=True, exist_ok=False)
    names = ["simple", "linear", "boosted_tree"]
    if fixture_predictors:
        names += ["learner_fixture", "tabpfn_fixture"]
    predictors: dict[str, str | ExternalPredictor] = {name: name for name in names}
    predictors.update(descriptors)
    names = list(predictors)
    results: list[Document] = []
    for name, predictor in predictors.items():
        paths = (saved_attempts or {}).get(name)
        if paths is not None and len(paths) != 3:
            raise ValueError("exactly three saved attempts required")
        for draw in (1, 2, 3):
            destination_path = destination / f"{name}-{draw}"
            if paths is None:
                result = attempt(
                    prepared,
                    destination_path,
                    predictor,
                    draw,
                    diagnostic=diagnostic,
                    cgroup_root=cgroup_root,
                )
            else:
                expected = cache_identity(prepared, predictor, draw, diagnostic=diagnostic)
                result = read_attempt(paths[draw - 1], expected)
                shutil.copytree(paths[draw - 1], destination_path)
            results.append(result)
    summaries: Document = {}
    for name in names:
        scores = [
            r["scores"] for r in results if r["predictor"] == name and r["status"] == "SUCCESS"
        ]
        if len(scores) == 3:
            summaries[name] = {key: summarize_draws([s[key] for s in scores]) for key in scores[0]}
    manifest = read_json(prepared / "manifest.json")
    report: Document = {
        "version": 1,
        "status": "INCOMPLETE",
        "evidence": "FIXTURE_ONLY",
        "dataset": manifest["dataset"],
        "partition": manifest["partition"],
        "task": manifest["task"],
        "summaries": summaries,
        "fixture_checks": "PASS" if len(summaries) == len(names) else "FAIL",
        "learner": "UNAVAILABLE: valid learner checkpoint required",
        "tabpfn": tabpfn_status(),
        "quality": "NOT_MEASURED",
        "resource_enforcement": "NOT_ENFORCED" if diagnostic else "see_attempt_receipts",
    }
    learner = "learner_fixture" if fixture_predictors else "learner"
    reference = "tabpfn_fixture" if fixture_predictors else "tabpfn"
    complete = all(
        name in summaries for name in ("simple", "linear", "boosted_tree", learner, reference)
    )
    gap: Document = {"status": "INCOMPLETE", "gap": None, "evidence": "FIXTURE_ONLY"}
    if complete:
        metric = "rmse" if manifest["task"] == "regression" else "brier"
        gap.update(
            comparison_gaps(
                manifest["task"],
                summaries[learner][metric]["draws"],
                summaries[reference][metric]["draws"],
                summaries["simple"][metric]["draws"],
            )
        )
    report["comparison"] = gap
    report["groups"] = aggregate_datasets(
        [
            {
                "dataset": manifest["dataset"],
                "task": manifest["task"],
                "partition": manifest["partition"],
                "comparison": gap,
            }
        ]
    )["groups"]
    report["quality"] = {"evidence": "FIXTURE_ONLY", "target_met": gap.get("quality_target_met")}
    report["attempts"] = [
        {
            "predictor": r["predictor"],
            "draw": r["draw"],
            "status": r["status"],
            "reason": r.get("reason"),
        }
        for r in results
    ]
    if learner in summaries:
        report["learner"] = "FIXTURE_ONLY: adapter results retained"
    if reference in summaries:
        report["tabpfn"] = {"status": "FIXTURE_ONLY", "real_inference": "NOT_RUN"}
    write_json(destination / "comparison.json", report)
    seal(destination)
    return report
