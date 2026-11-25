"""Subprocess entrypoint used by run_guarded; configuration arrives over stdin."""

import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import numpy as np
import torch

from r0.modules.generation import RunRequest, run_generation
from r0.modules.priors import TrainingTask, generate_task, recipe_from_dict


def main() -> int:
    data = json.load(sys.stdin)
    config = data["request"]
    recipe = config.pop("recipe")
    request = RunRequest(recipe_from_dict(recipe), **config)
    sync_path: str | None = data["sync_path"]
    generated = 0
    counter = Lock()
    if request.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        torch.cuda.reset_peak_memory_stats()

    def produce(task_id: int) -> TrainingTask:
        nonlocal generated
        task = generate_task(
            request.recipe, seed=request.seed, task_id=task_id, device=request.device
        )
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
    print(json.dumps(report, sort_keys=True))
    return 0 if result.status == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
