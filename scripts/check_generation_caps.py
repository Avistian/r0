"""Validate a generation dispatch before the credential-bearing workflow step."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from check_device_agreement import agreement_matrix

from r0.modules.generation import load_request
from r0.modules.priors import CompositeRecipe, LinearRecipe, sample_recipe


def positive_limit(name: str) -> float:
    value = float(os.environ[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def agreement_recipe(recipe, task_type, classes):
    """Keep the mechanism settings while making parent and child outputs compatible."""
    controls = recipe if isinstance(recipe, LinearRecipe) else recipe.task
    task = replace(
        controls,
        rows=64,
        context_rows=48,
        task_type=task_type,
        classes=classes,
        separation=1.0,
        class_offsets=(),
        noise_ratio=0.0,
    )
    if isinstance(recipe, LinearRecipe):
        return task
    if isinstance(recipe, CompositeRecipe):
        components = tuple(
            replace(component, recipe=agreement_recipe(component.recipe, task_type, classes))
            for component in recipe.components
        )
        return replace(recipe, task=task, components=components)
    return replace(recipe, task=task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]{3,40}", args.run):
        raise ValueError("invalid run name")
    if not args.config.resolve().is_relative_to(Path("configs/generation").resolve()):
        raise ValueError("request must be inside configs/generation")
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be 1..16")
    if os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise ValueError("do not rerun an uncertain submission; use a new explicit dispatch")
    request = load_request(args.config)
    for value, variable in (
        (request.count, "MAX_GENERATION_TASKS"),
        (request.max_seconds, "MAX_GENERATION_SECONDS"),
        (request.max_bytes, "MAX_GENERATION_BYTES"),
    ):
        if value > positive_limit(variable):
            raise ValueError(f"request exceeds {variable}")
    job_minutes = float(os.environ.get("GENERATION_JOB_TIMEOUT_MINUTES", "15"))
    if not math.isfinite(job_minutes) or not 1 <= job_minutes <= 75:
        raise ValueError("generation job timeout must be 1..75 minutes")
    agreement_seconds = 180 if request.device == "cuda" else 0
    if 2 * (request.max_seconds + 180 + agreement_seconds) > job_minutes * 60:
        raise ValueError("generation plus resume and startup exceed the workflow deadline")
    gpu = "T4" if request.device == "cuda" else "none"
    if gpu != "none" and os.environ.get("MAX_GPU") != "T4":
        raise ValueError("GPU generation requires MAX_GPU=T4")
    history = json.loads(args.history.read_text())
    if len(history) >= 1000:
        raise ValueError("history is possibly truncated; do not infer remaining budget")
    today = dt.datetime.now(dt.UTC).date().isoformat()
    today_runs = {str(r["databaseId"]) for r in history if r["createdAt"].startswith(today)}
    today_runs.add(os.environ["GITHUB_RUN_ID"])
    daily_runs = positive_limit("MAX_TRAIN_RUNS_PER_DAY")
    reservation = positive_limit("GENERATION_RUN_RESERVATION_USD")
    if len(today_runs) > daily_runs:
        raise ValueError("daily workflow dispatch limit reached, including failed dispatches")
    if daily_runs * reservation > positive_limit("GENERATION_DAILY_BUDGET_USD"):
        raise ValueError("daily reservations exceed the authorized budget")
    # Two calls (generation + independent-container resume); conservative 2x allowance.
    rate = 0.0000131 + 4 * 0.00000222 + (0.000164 if gpu == "T4" else 0)
    estimated = 2 * (request.max_seconds + 180 + agreement_seconds) * rate
    if estimated * 2 + 0.25 > reservation:
        raise ValueError("resource estimate plus build allowance exceeds run reservation")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != os.environ["GITHUB_SHA"] or not re.fullmatch(r"[a-f0-9]{40}", source):
        raise ValueError("checkout differs from dispatched immutable source")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], check=True)
    agreement_cases = []
    for task_type, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
        diagnostic = agreement_recipe(request.recipe, task_type, classes)
        if request.sampling is not None:
            agreement = sample_recipe(
                request.sampling.family,
                profile=request.sampling.profile,
                task=diagnostic,
                seed=request.seed,
                task_id=request.start,
            )
        else:
            agreement = diagnostic
        agreement_cases.append({"name": f"requested-{task_type}", "recipe": asdict(agreement)})
    if gpu != "none":
        agreement_cases.extend(agreement_matrix())
    plan = {
        "source": source,
        "run": args.run,
        "workflow_run_id": os.environ["GITHUB_RUN_ID"],
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "request": asdict(request),
        "agreement_cases": agreement_cases,
        "agreement_seconds": agreement_seconds,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "workers": args.workers,
        "retry_failed": args.retry_failed,
        "gpu": gpu,
        "cpu_limit": 1,
        "memory_limit_mib": 4096,
        "reserved_usd": reservation,
        "estimated_function_usd": estimated,
        "actual_cost_usd": "NOT_MEASURED",
        "pricing_source": "https://modal.com/pricing",
        "status": "RESERVED_NOT_SUBMITTED",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
