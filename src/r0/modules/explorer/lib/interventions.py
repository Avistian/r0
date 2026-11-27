"""Retained bounded intervention attempts; successful results never replace the baseline."""

import base64
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from r0.modules.explorer.lib.comparisons import SavedPair
from r0.modules.explorer.lib.construction import caller_limits
from r0.modules.explorer.lib.saved import inside, retain_record
from r0.modules.generation import load_group, run_guarded_document
from r0.modules.priors import TrainingTask


def intervention_controls(task: TrainingTask) -> dict[str, Any]:
    original = json.loads(task.recipe_json)["recipe"]
    version = original["version"]
    runtime = original.get("runtime", original)
    if "graph" in original:
        runtime = original["graph"]["runtime"]
    if version in (
        "sampled-native-graph-v1",
        "sampled-native-vector-graph-v1",
        "sampled-converted-graph-v1",
    ):
        runtime = original["recipe"]
    if runtime.get("version") not in (
        "functional-causal-v1",
        "standalone-network-innovations-v1",
        "functional-causal-v2",
        "fitted-functional-causal-v1",
        "sampled-fitted-functional-causal-v1",
        "population-fitted-functional-causal-v1",
        "sampled-population-fitted-functional-causal-v1",
        "support-population-functional-causal-v1",
        "native-causal-v1",
        "native-vector-causal-v1",
        "converted-causal-v1",
    ):
        raise ValueError("causal interventions require a frozen functional or native graph")
    native = runtime["version"] in ("native-causal-v1", "native-vector-causal-v1")
    converted = runtime["version"] == "converted-causal-v1"
    vector = runtime["version"] == "native-vector-causal-v1"
    nodes: list[dict[str, Any]] = []
    for i, node in enumerate(runtime["nodes"]):
        equation = node["equation"] if native or converted else node
        domain = node.get("domain")
        pipeline: dict[str, Any] = equation.get("pipeline") or {}
        nodes.append(
            {
                "index": i,
                "parents": equation["parents"],
                "domain": domain,
                "domains": node.get("domains"),
                "converters": node.get("converters", []),
                "root": equation.get("root"),
                "pipeline": equation.get("pipeline"),
                "function": equation["function"],
                "noise_scale": equation["noise_scale"],
                "noise_response": equation["noise_response"],
                "exogenous": equation["exogenous"],
                "captured_channels": [c["channel"] for c in pipeline.get("converters", [])],
            }
        )
    return {
        "nodes": nodes,
        "native": native,
        "vector": vector,
        "converted": converted,
        "version": version,
        "notice": (
            "Original construction and fit archives remain unchanged. "
            "Interventions use frozen calibration. CPU only; partial coupled mechanisms "
            "are evaluated before clamping. Vector clamps select logical variables; "
            "all-variable clamps bypass the joint equation. Root/domain/pipeline changes "
            "retain original archive and geometry. Converter captures precede structural noise; "
            "capture-only clamps keep descendant representation, representation clamps retain "
            "later noise and importance. Original-input converters "
            "have no unique inverse representation."
        ),
    }


def _failure_detail(stderr: str, *, timed_out: bool) -> str:
    if timed_out:
        return "The time limit was exceeded. Increase it or reduce the request."
    # Full worker diagnostics stay in the result receipt; only validation messages
    # belong in the primary user-facing banner.
    last = next((line.strip() for line in reversed(stderr.splitlines()) if line.strip()), "")
    for prefix in ("ValueError: ", "json.decoder.JSONDecodeError: "):
        if last.startswith(prefix):
            return last[len(prefix) :][:240]
    return "Check the intervention settings and limits. Details are retained in the attempt record."


def guarded_intervention(
    output: Path,
    operation: str,
    data: dict[str, Any],
    *,
    before: TrainingTask | None = None,
    archive: str | None = None,
    pair_path: str | None = None,
) -> dict[str, Any]:
    attempts = output / ".intervention-attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    request: dict[str, Any] = {
        "causal_action": operation,
        "max_seconds": data.get("max_seconds", 60),
        "max_bytes": data.get("max_bytes", 256 * 1024**2),
    }
    pending = {
        "operation": operation,
        "before_episode_id": before.episode_id if before else None,
        "data": data,
    }
    retain_record(attempts / f"{identity}-started.json", {**pending, "status": "STARTED"})
    try:
        request["causal_limits"] = asdict(caller_limits(data))
        if data.get("device", "cpu") != "cpu":
            raise ValueError("causal intervention workflows support explicit CPU only")
        if operation in ("create", "sample_policy"):
            if before is None or archive is None:
                raise ValueError("save an original dataset before intervening")
            request.update(
                before_archive=str(inside(output, output / archive)),
                before_episode_id=before.episode_id,
                interventions_json=data.get("interventions_json", "[]"),
                policy_json=data.get("policy_json", "{}"),
                mode=data.get("mode", "matched"),
                population=data.get("population", 1),
            )
        else:
            request["pair_path"] = str(inside(output, output / (pair_path or data["path"])))
            if operation == "inspect":
                request["indices"] = data["indices"]
        run = run_guarded_document(request, output / "interventions" / identity)
        failed = run.timed_out or run.returncode or run.summary is None
        retain_record(
            attempts / f"{identity}-result.json",
            {**pending, "status": "FAILED" if failed else "COMPLETE", "result": asdict(run)},
        )
        if failed:
            raise ValueError(
                f"Intervention {operation} failed; previous dataset/comparison retained. "
                f"{_failure_detail(run.stderr, timed_out=run.timed_out)}"
            )
        return dict(run.summary or {})
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, KeyboardInterrupt) as error:
        retain_record(
            attempts / f"{identity}-failed.json",
            {**pending, "status": "FAILED", "error": repr(error)},
        )
        raise


def saved_intervention(output: Path, result: dict[str, Any]) -> SavedPair:
    path = inside(output, Path(result["pair_path"]))
    before = next(load_group(path / "before.zip"))
    after = next(load_group(path / "after.zip"))
    return SavedPair(
        before,
        after,
        {
            "card": result["card"],
            "before_archive": str((path / "before.zip").relative_to(output)),
            "after_archive": str((path / "after.zip").relative_to(output)),
        },
        str(path.relative_to(output)),
    )


def intervention_download(output: Path, result: dict[str, Any]) -> dict[str, str]:
    path = inside(output, Path(result["download"]))
    return {
        "name": "causal-intervention-pair.zip",
        "mime": "application/zip",
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
    }
