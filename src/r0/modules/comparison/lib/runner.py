"""Immutable attempts, explicit partial comparisons and exact-identity cache reads."""

import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .execution import environment, run_process
from .predictors import Array, Unsupported, predict, recipe
from .scoring import check_invariance, score
from .storage import Document, digest, file_hash, read_json, seal, verify, write_json
from .tabpfn import tabpfn_status


def worker(input_path: Path, output: Path, predictor: str) -> None:
    """Versioned input-only worker; never receives a scoring path."""
    view = read_json(input_path)
    start = time.monotonic()
    prediction, evidence = predict(view, predictor)
    reverse = list(reversed(range(len(view["query_ids"]))))
    reordered = {
        **view,
        "query_ids": [view["query_ids"][i] for i in reverse],
        "query_values": [view["query_values"][i] for i in reverse],
    }
    reversed_prediction, _ = predict(reordered, predictor)
    chunks: list[Array] = []
    for i in range(len(view["query_ids"])):
        chunk = {
            **view,
            "query_ids": view["query_ids"][i : i + 1],
            "query_values": view["query_values"][i : i + 1],
        }
        result, _ = predict(chunk, predictor)
        chunks.append(result)
    check_invariance(prediction, reversed_prediction[reverse], np.concatenate(chunks))
    output.mkdir(exist_ok=False)
    np.save(output / "predictions.npy", prediction, allow_pickle=False)
    write_json(
        output / "prediction.json",
        {
            "version": 1,
            "query_ids": view["query_ids"],
            "evidence": evidence,
            "invariance": "PASS",
            "seconds": time.monotonic() - start,
            "environment": environment(),
        },
    )


def cache_identity(prepared: Path, predictor: str, draw: int, *, diagnostic: bool) -> Document:
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
        "predictor": recipe(predictor, manifest["task"]),
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
    predictor: str,
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
        "predictor": predictor,
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
        if diagnostic:
            worker(destination / "input.json", destination / "output", predictor)
            result["resources"] = {
                "status": "NOT_ENFORCED",
                "reason": "explicit fixture diagnostic",
            }
        else:
            if cgroup_root is None:
                raise ValueError("enforced attempt requires delegated cgroup_root")
            receipt = run_process(
                [
                    sys.executable,
                    "-m",
                    "r0.modules.comparison",
                    "worker",
                    "--input",
                    str((destination / "input.json").resolve()),
                    "--out",
                    str((destination / "output").resolve()),
                    "--predictor",
                    predictor,
                ],
                cwd=destination.resolve(),
                cgroup_root=cgroup_root,
            )
            result["resources"] = receipt
            if receipt["status"] != "COMPLETED":
                raise ValueError(f"predictor process {receipt['status']}")
        shutil.copyfile(prepared / "scoring.json", destination / "scoring.json")
        labels = read_json(destination / "scoring.json")
        output = read_json(destination / "output/prediction.json")
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
    except (OSError, ValueError, RuntimeError) as error:
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
) -> Document:
    verify(prepared)
    destination.mkdir(parents=True, exist_ok=False)
    names = ["simple", "linear", "boosted_tree"]
    if fixture_predictors:
        names += ["learner_fixture", "tabpfn_fixture"]
    results = [
        attempt(
            prepared,
            destination / f"{name}-{draw}",
            name,
            draw,
            diagnostic=diagnostic,
            cgroup_root=cgroup_root,
        )
        for name in names
        for draw in (1, 2, 3)
    ]
    summaries: Document = {}
    for name in names:
        scores = [
            r["scores"] for r in results if r["predictor"] == name and r["status"] == "SUCCESS"
        ]
        if len(scores) == 3:
            summaries[name] = {
                key: {
                    "draws": [s[key] for s in scores],
                    "mean": float(np.mean([s[key] for s in scores])),
                    "min": min(s[key] for s in scores),
                    "max": max(s[key] for s in scores),
                }
                for key in scores[0]
            }
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
    write_json(destination / "comparison.json", report)
    seal(destination)
    return report
