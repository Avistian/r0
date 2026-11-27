"""Bounded CPU measurements; each case runs in a fresh process, without retries."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CASES = tuple(
    (family, 1024, 8)
    for family in (
        "dense",
        "individual-sparse",
        "block-sparse",
        "skip",
        "residual",
        "irregular",
        "gated",
    )
) + (("dense", 20000, 8), ("dense", 16, 256))
PART_BYTES = 900_000


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def measure(index: int, output: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    from r0.modules import priors as p
    from r0.modules.generation import load_group, save_group

    torch.set_num_threads(1)
    family, rows, features = CASES[index]
    result: dict[str, Any] = {
        "case": index,
        "family": family,
        "rows": rows,
        "features": features,
        "seed": 28,
        "task_id": 9007199254740993,
        "status": "STARTED",
        "torch_threads": torch.get_num_threads(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    start = time.perf_counter()
    try:
        request = p.NetworkSamplingRequest(features, (8, 8, 1), families=(family,))
        network = p.sample_network(
            request, named_rng=np.random.default_rng(28), raw_rng=np.random.default_rng(29)
        )
        recipe = p.NetworkRecipe(
            p.LinearRecipe(
                rows=rows,
                context_rows=rows // 2,
                features=features,
                signal_features=features,
                active_features=features,
            ),
            network,
        )
        result["admission"] = recipe.runtime.admit()
        result["construction_seconds"] = time.perf_counter() - start
        start_generate = time.perf_counter()
        task = p.generate_task(recipe, seed=28, task_id=9007199254740993)
        result["generation_seconds"] = time.perf_counter() - start_generate
        assert all(bool(torch.isfinite(t).all()) for t in (task.inputs, task.scores, task.targets))
        result["episode_id"] = task.episode_id
        result["recipe_sha256"] = hashlib.sha256(task.recipe_json.encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="r0-network-measure-") as temp:
            archive = Path(temp) / "task.zip"
            save_start = time.perf_counter()
            receipt = save_group(archive, [(9007199254740993, task)], max_bytes=64 * 1024**2)
            result["save_seconds"] = time.perf_counter() - save_start
            load_start = time.perf_counter()
            reopened = next(load_group(archive))
            result["load_seconds"] = time.perf_counter() - load_start
            assert reopened.recipe_json == task.recipe_json
            for field in ("inputs", "targets", "scores", "context_indices", "query_indices"):
                assert torch.equal(getattr(task, field), getattr(reopened, field)), field
            replay_start = time.perf_counter()
            record = json.loads(reopened.recipe_json)
            replay = p.generate_task(
                p.recipe_from_dict(record["recipe"]),
                seed=record["seed"],
                task_id=record["task_id"],
            )
            result["regeneration_seconds"] = time.perf_counter() - replay_start
            assert replay.recipe_json == task.recipe_json
            for field in ("inputs", "targets", "scores", "context_indices", "query_indices"):
                assert torch.equal(getattr(task, field), getattr(replay, field)), field
            result["archive_bytes"] = receipt.size_bytes
            result["archive_sha256"] = receipt.sha256
            parts: list[dict[str, Any]] = []
            with archive.open("rb") as stream:
                while chunk := stream.read(PART_BYTES):
                    name = f"task.zip.part{len(parts):04d}"
                    (output / name).write_bytes(chunk)
                    parts.append(
                        {
                            "name": name,
                            "bytes": len(chunk),
                            "sha256": hashlib.sha256(chunk).hexdigest(),
                        }
                    )
            result["archive_parts"] = parts
            (output / "task-receipt.json").write_bytes(archive.with_suffix(".json").read_bytes())
        result["status"] = "PASS"
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    result["case_seconds"] = time.perf_counter() - start
    # Linux ru_maxrss is KiB. Includes Python/library startup, retained tasks,
    # save/load and evidence copying; it is not the admission tensor estimate.
    result["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", type=int, choices=range(len(CASES)))
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("RSS accounting currently supports Linux only")
    output = args.output.resolve()
    if args.case is not None:
        write_json(output / "result.json", measure(args.case, output))
        return 0
    root = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root / "docs/evidence"):
        parser.error("output must be inside this checkout's ignored docs/evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    identity = {
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_diff": subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src"], cwd=root, text=True
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "started_epoch": time.time(),
        "case_count": len(CASES),
        "protocol": "one fresh child per fixed case; no filtering/retries; CPU only",
        "qualification": (
            "shared host; process RSS includes imports; no GPU or throughput extrapolation"
        ),
    }
    write_json(output / "started.json", identity)
    results: list[dict[str, Any]] = []
    for index in range(len(CASES)):
        directory = output / f"case-{index:02d}"
        directory.mkdir()
        with (directory / "worker.log").open("x") as log:
            try:
                run = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        str(directory),
                        "--case",
                        str(index),
                    ],
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
                state = "EXITED" if run.returncode == 0 else "PROCESS_FAILED"
                code = run.returncode
            except subprocess.TimeoutExpired:
                state, code = "TIMED_OUT", None
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {"status": state}
        result.update(process_status=state, returncode=code)
        results.append(result)
        write_json(directory / "process.json", {"status": state, "returncode": code})
        print(index, result["status"], flush=True)
    write_json(
        output / "summary.json", {**identity, "finished_epoch": time.time(), "results": results}
    )
    return int(
        any(
            result["status"] != "PASS"
            or result["process_status"] != "EXITED"
            or result["returncode"] != 0
            for result in results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
