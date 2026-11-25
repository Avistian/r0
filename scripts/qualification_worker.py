"""One bounded measurement process; imports ML libraries after setting its memory limit."""

import argparse
import json
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--address-space-bytes", type=int, required=True)
    parser.add_argument("--probe", choices=("memory", "timeout"))
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (args.address_space_bytes, args.address_space_bytes))
    assert resource.getrlimit(resource.RLIMIT_AS) == (args.address_space_bytes,) * 2
    if args.probe == "memory":
        try:
            bytearray(args.address_space_bytes + 1)
        except MemoryError:
            (args.directory / "worker.json").write_text('{"memory_limit": "ENFORCED"}\n')
            return
        raise AssertionError("address-space limit did not reject allocation")
    if args.probe == "timeout":
        (args.directory / "started.json").write_text('{"sleep_seconds": 60}\n')
        time.sleep(60)
        raise AssertionError("watchdog failed to interrupt sleep")

    import numpy as np
    import torch

    from r0.modules.generation import load_group, request_from_dict, run_generation
    from r0.modules.priors import explain_generation, generate_task

    request = request_from_dict(json.loads((args.directory / "request.json").read_text()))
    assert torch.get_num_threads() == 1
    started = time.perf_counter()
    summary = run_generation(request, args.directory / "run", workers=1)
    seconds = time.perf_counter() - started
    result = {
        "summary": asdict(summary),
        "generation_and_save_seconds": seconds,
        "generation_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "address_space_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
        "hardware": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "verification": "NOT_RUN",
    }
    # Publish the timing before verification; a verification timeout must not erase it.
    (args.directory / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    if summary.status != "COMPLETE":
        raise RuntimeError(f"generation did not complete: {summary}")
    archives = list((args.directory / "run").rglob("data-*.zip"))
    assert len(archives) == 1
    saved = list(load_group(archives[0]))
    assert len(saved) == 1
    task = saved[0]
    replay = generate_task(request.recipe, seed=request.seed, task_id=request.start)
    for name in ("inputs", "scores", "targets", "context_indices", "query_indices"):
        assert torch.equal(getattr(task, name), getattr(replay, name)), name
    if task.probabilities is not None:
        assert torch.equal(task.probabilities, replay.probabilities)
    graph = explain_generation(task, [0, 1, task.inputs.shape[0] - 1])
    assert graph["target_max_error"] <= 1e-9
    assert graph["signal_max_error"] <= 1e-9
    assert torch.isfinite(task.inputs).all()
    assert torch.isfinite(task.scores).all()
    assert torch.isfinite(task.targets).all()
    result.update(
        {
            "verification": "PASS",
            "graph_target_error": graph["target_max_error"],
            "graph_signal_error": graph["signal_max_error"],
            "episode_id": task.episode_id,
            "archive_bytes": archives[0].stat().st_size,
            "verification_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }
    )
    (args.directory / "worker.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
