"""Retain repeated CPU generation measurements, bounded failures and scaling estimates."""

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix(".pending")
    with pending.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def execute(directory, seconds, memory_bytes, probe=None):
    command = [
        sys.executable,
        str(Path(__file__).with_name("qualification_worker.py")),
        "--directory",
        str(directory),
        "--address-space-bytes",
        str(memory_bytes),
    ]
    if probe:
        command += ["--probe", probe]
    write(directory / "launch-request.json", {"command": command, "wall_seconds": seconds})
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    timed_out = False
    try:
        write(directory / "launch.json", {"pid": process.pid})
        stdout, stderr = process.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    result = {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "process_seconds": time.perf_counter() - started,
    }
    result["receipt_errors"] = {}
    for name in ("measurement", "worker"):
        path = directory / f"{name}.json"
        result[name] = None
        if path.exists():
            try:
                result[name] = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                result["receipt_errors"][name] = str(error)
    result["status"] = (
        "TIMEOUT"
        if timed_out
        else ("PASS" if process.returncode == 0 and result["worker"] is not None else "FAILED")
    )
    write(directory / "result.json", result)
    return result


def summarize(root):
    manifest = json.loads((root / "manifest.json").read_text())
    groups = {}
    totals = {
        "PASS": 0,
        "FAILED": 0,
        "TIMEOUT": 0,
        "INCOMPLETE": 0,
        "LAUNCH_UNKNOWN": 0,
        "NOT_RUN": 0,
    }
    for case in manifest["cases"]:
        path = root / case["id"] / "result.json"
        launched = (path.parent / "launch.json").exists() or (
            path.parent / "worker-started.json"
        ).exists()
        missing = (
            "INCOMPLETE"
            if launched
            else ("LAUNCH_UNKNOWN" if (path.parent / "launch-request.json").exists() else "NOT_RUN")
        )
        result = json.loads(path.read_text()) if path.exists() else {"status": missing}
        status = result["status"]
        if status == "PASS" and result["worker"].get("verification") != "PASS":
            status = "FAILED"
        totals[status] += 1
        key = (case["family"], case["size"], case["task_type"])
        groups.setdefault(key, []).append((case, result, status))
    rows = []
    for key, entries in groups.items():
        successes = [result["worker"] for _, result, status in entries if status == "PASS"]
        qualified = len(successes) == len(entries) and len(entries) >= 3
        times = [item["generation_and_save_seconds"] for item in successes]
        recipe = entries[0][0]["request"]["recipe"]
        controls = recipe.get("task", recipe)
        rows.append(
            {
                "family": key[0],
                "size": key[1],
                "task_type": key[2],
                "rows": controls["rows"],
                "features": controls["features"],
                "active_features": controls["active_features"],
                "classes": controls["classes"],
                "planned": len(entries),
                "confirmed_launched": sum(
                    s not in ("NOT_RUN", "LAUNCH_UNKNOWN") for _, _, s in entries
                ),
                "launch_unknown": sum(s == "LAUNCH_UNKNOWN" for _, _, s in entries),
                "passed": len(successes),
                "outcomes": {status: sum(s == status for _, _, s in entries) for status in totals},
                "qualification": "MEASURED_LOCAL_CPU" if qualified else "NOT_QUALIFIED",
                "seconds_min_median_max": [min(times), statistics.median(times), max(times)]
                if times
                else None,
                "generation_peak_rss_bytes": max(
                    (s["generation_peak_rss_bytes"] for s in successes), default=None
                ),
                "extrapolated_100m_generation_days": statistics.median(times) * 100_000_000 / 86400
                if qualified
                else None,
                "extrapolated_100m_archive_bytes": statistics.median(
                    [s["archive_bytes"] for s in successes]
                )
                * 100_000_000
                if qualified
                else None,
                "case_ids": [case["id"] for case, _, _ in entries],
            }
        )
    return {
        "source": manifest["source"],
        "suite": manifest["suite"],
        "totals": totals,
        "cases": rows,
        "estimate_scope": (
            "single CPU worker, fixed settings, one-task archives; excludes startup, verification, "
            "failures, retries and platform overhead; no production mixture selected"
        ),
        "estimated_cost": "NOT_MEASURED",
        "gpu": "NOT_RUN",
        "hosted": "NOT_RUN",
    }


def main():
    from qualification_cases import FAMILIES, SHAPES, TASK_TYPES, cases

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--sizes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--task-types", nargs="+", choices=TASK_TYPES, default=TASK_TYPES)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--max-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--address-space-gib", type=int, default=8)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.out), indent=2, allow_nan=False))
        return
    if not 1 <= args.repetitions <= 10 or not 2 <= args.address_space_gib <= 16:
        parser.error("repetitions must be 1..10 and address space 2..16 GiB")
    if not math.isfinite(args.max_seconds) or not 1 <= args.max_seconds <= 600:
        parser.error("per-case wall seconds must be 1..600")
    if not 65536 <= args.max_bytes <= 1024**3:
        parser.error("per-case output bytes must be 64 KiB..1 GiB")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src", "scripts"], check=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    drivers = {}
    for name in ("qualify_generators.py", "qualification_cases.py", "qualification_worker.py"):
        path = Path(__file__).with_name(name)
        committed = subprocess.check_output(["git", "show", f"HEAD:scripts/{name}"])
        assert committed == path.read_bytes(), f"commit measurement driver: {name}"
        drivers[name] = hashlib.sha256(committed).hexdigest()
    selected = cases(
        args.families,
        args.sizes,
        args.task_types,
        args.repetitions,
        args.max_seconds,
        args.max_bytes,
    )
    if len({case["id"] for case in selected}) != len(selected):
        parser.error("do not repeat family, size or task-type arguments")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "suite": "fixed-cpu-qualification-v1",
        "source": source,
        "drivers": drivers,
        "address_space_bytes": args.address_space_gib * 1024**3,
        "cases": selected,
        "cpu_info": Path("/proc/cpuinfo").read_text(),
        "mem_info": Path("/proc/meminfo").read_text(),
        "cpu_quota": "NOT_ENFORCED",
        "threads": 1,
    }
    write(args.out / "manifest.json", manifest)
    for probe in ("memory", "timeout"):
        directory = args.out / f"probe-{probe}"
        directory.mkdir()
        result = execute(directory, 5 if probe == "memory" else 1, 128 * 1024**2, probe)
        if probe == "memory":
            assert result["worker"] == {"memory_limit": "ENFORCED"}, result
        else:
            assert result["timed_out"], result
            assert (directory / "started.json").exists(), "probe never started"
            assert "worker" in result["receipt_errors"], "damaged receipt was not retained"
    for index, case in enumerate(selected):
        directory = args.out / case["id"]
        directory.mkdir()
        write(directory / "request.json", case["request"])
        result = execute(directory, args.max_seconds, manifest["address_space_bytes"])
        print(f"{index + 1}/{len(selected)} {case['id']}: {result['status']}", flush=True)
    report = summarize(args.out)
    write(args.out / "report.json", report)
    print(json.dumps(report["totals"]), flush=True)


if __name__ == "__main__":
    main()
