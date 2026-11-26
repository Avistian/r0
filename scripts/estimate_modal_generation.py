"""Estimate a larger request from a retained Modal result; never launch compute."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import load_request, request_from_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result_bytes = args.result.read_bytes()
    result = json.loads(result_bytes)
    plan = result["plan"]
    generation = result["generation"]
    measured = generation["summary"]
    baseline = request_from_dict(plan["request"])
    proposed = load_request(args.config)
    if generation["source"] != plan["source"] or measured["device"] != baseline.device:
        raise ValueError("measurement source or device differs from its request")
    if (
        result["status"] != "COMPLETE"
        or generation["returncode"] != 0
        or generation["timed_out"]
        or measured["status"] != "COMPLETE"
        or measured["failed"] != 0
        or measured["pending"] != 0
        or not measured["generated"]
        == measured["attempted"]
        == measured["completed"]
        == baseline.count
    ):
        raise ValueError("use a complete fresh generation result; not a resume or partial run")
    seconds = measured["generation_seconds"]
    rate = measured["generated_per_second"]
    if not (math.isfinite(seconds) and seconds > 0 and math.isfinite(rate) and rate > 0):
        raise ValueError("measurement duration and rate must be finite and positive")
    if not math.isclose(rate, baseline.count / seconds, rel_tol=1e-9):
        raise ValueError("recorded rate disagrees with measured count and time")
    expected, actual = asdict(baseline), asdict(proposed)
    for name in ("count", "max_seconds", "max_bytes"):
        expected.pop(name)
        actual.pop(name)
    if expected != actual:
        raise ValueError(
            "recipe, sampling, seed, start, grouping and device must match the measurement"
        )
    estimate = proposed.count / rate
    report = {
        "baseline_source": plan["source"],
        "baseline_workflow_run_id": plan["workflow_run_id"],
        "baseline_hardware": measured["hardware"],
        "baseline_software": {
            name: measured.get(name) for name in ("python", "numpy", "torch", "cuda")
        },
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "workers": plan["workers"],
        "measured_tasks": measured["generated"],
        "measured_tasks_per_second": rate,
        "measured_generation_seconds": seconds,
        "measured_peak_rss_bytes": measured["peak_rss_bytes"],
        "measured_gpu_peak_allocated_bytes": measured["gpu_peak_allocated_bytes"],
        "requested_tasks": proposed.count,
        "estimated_generation_seconds": estimate,
        "generation_time_limit_seconds": proposed.max_seconds,
        "fits_generation_time_limit": estimate <= proposed.max_seconds,
        "output_limit_bytes": proposed.max_bytes,
        "estimated_output_bytes": "NOT_MEASURED",
        "scope": "All requested tasks treated as new; "
        "assumes unchanged generator, hardware and workers. "
        "Excludes startup, agreement checks, resume verification and retries. "
        "Sampled settings vary; this is a projection from the measured tasks, "
        "not profile qualification. "
        "This command does not grant compute or budget admission.",
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
