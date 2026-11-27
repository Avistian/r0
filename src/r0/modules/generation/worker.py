"""Subprocess entrypoint used by run_guarded; configuration arrives over stdin."""

import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock

import numpy as np
import torch

from r0.modules.generation import load_group, run_generation
from r0.modules.generation.lib.requests import generate_requested, request_from_dict
from r0.modules.priors import TrainingTask, compare_pair, paired_recipe


def main() -> int:
    data = json.load(sys.stdin)
    raw_request = dict(data["request"])
    if "causal_action" in raw_request:
        from r0.modules.generation.lib.causal_jobs import causal_job

        print(json.dumps(causal_job(raw_request, Path(data["out"])), sort_keys=True))
        return 0
    pairing = raw_request.pop("pairing", None)
    request = request_from_dict(raw_request)
    if pairing is not None:
        request = replace(
            request,
            recipe=paired_recipe(
                request.recipe, intervention=pairing["intervention"], value=pairing["value"]
            ),
        )
    sync_path: str | None = data["sync_path"]
    generated = 0
    counter = Lock()
    if request.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        torch.cuda.reset_peak_memory_stats()

    def produce(task_id: int) -> TrainingTask:
        nonlocal generated
        task = generate_requested(request, task_id)
        with counter:
            generated += 1
        return task

    def persist() -> None:
        if sync_path is not None:
            subprocess.run(["sync", sync_path], check=True, capture_output=True)

    start = time.monotonic()
    result = run_generation(
        request,
        Path(data["out"]),
        workers=data["workers"],
        retry_failed=data["retry_failed"],
        persist=persist,
        produce=produce,
    )
    persist()
    seconds = time.monotonic() - start
    report = {
        **asdict(result),
        "generated": generated,
        "generation_seconds": seconds,
        "generated_per_second": generated / seconds,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "gpu_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if request.device == "cuda" else None
        ),
        "device": request.device,
        "hardware": torch.cuda.get_device_name()
        if request.device == "cuda"
        else platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if pairing is not None and result.status == "COMPLETE":
        before = next(
            task
            for task in load_group(Path(pairing["before_archive"]))
            if json.loads(task.recipe_json)["task_id"] == request.start
        )
        archives = list(Path(data["out"]).glob("groups/**/*.zip"))
        if len(archives) != 1:
            raise ValueError("paired generation requires one completed archive")
        after = next(
            task
            for task in load_group(archives[0])
            if json.loads(task.recipe_json)["task_id"] == request.start
        )
        report["pair_card"] = compare_pair(
            before,
            after,
            intervention=pairing["intervention"],
            allow_estimated_variance=pairing["allow_estimated_variance"],
            limits=request.causal_limits,
        )
    print(json.dumps(report, sort_keys=True))
    return 0 if result.status == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
