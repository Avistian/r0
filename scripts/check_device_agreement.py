"""Compare named generation cases on explicit devices; keep every attempted outcome."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path


def agreement_matrix():
    """Fixed small/medium/large representatives; no sampling or fallback selection."""
    from dataclasses import asdict, replace

    from r0.modules.priors import (
        Combination,
        CompositeRecipe,
        FourierRecipe,
        FunctionComponent,
        GaussianProcessRecipe,
        GroupedLinearRecipe,
        LinearRecipe,
        PolynomialRecipe,
        QuadraticRecipe,
        SmoothRecipe,
    )

    cases = []
    for size, rows, features, active in (
        ("small", 64, 8, 4),
        ("medium", 1024, 32, 16),
        ("large", 20000, 256, 256),
    ):
        for kind, classes in (
            ("regression", 0),
            ("binary", 2),
            ("multiclass", 100 if size == "large" else 3),
        ):
            task = LinearRecipe(
                rows=rows,
                context_rows=rows * 3 // 4,
                features=features,
                signal_features=features,
                active_features=active,
                task_type=kind,
                classes=classes,
            )
            periodic = replace(task, active_features=min(active, 64))
            recipes = {
                "linear": task,
                "grouped": GroupedLinearRecipe(task=task, group_size=4, calibration_rows=1024),
                "polynomial": PolynomialRecipe(task=task, degree=4, calibration_rows=1024),
                "quadratic": QuadraticRecipe(task=task, rank=32, calibration_rows=1024),
                "rff": FourierRecipe(task=task, family="rff", terms=256, calibration_rows=1024),
                "periodic": FourierRecipe(
                    task=periodic, family="periodic", terms=64, calibration_rows=1024
                ),
                "radial": SmoothRecipe(
                    task=task, family="radial", terms=max(64, active), calibration_rows=1024
                ),
                "sinusoidal": SmoothRecipe(
                    task=task, family="sinusoidal", terms=max(64, active), calibration_rows=1024
                ),
            }
            components = []
            for recipe in recipes.values():
                controls = recipe if isinstance(recipe, LinearRecipe) else recipe.task
                child_task = replace(controls, rows=16, context_rows=8)
                child = (
                    child_task
                    if isinstance(recipe, LinearRecipe)
                    else replace(
                        recipe,
                        task=child_task,
                        calibration_rows=256,
                    )
                )
                axes = (
                    tuple(range(controls.active_features))
                    if controls.active_features != active
                    else ()
                )
                components.append(FunctionComponent(child, axes=axes))
            recipes["composition"] = CompositeRecipe(
                task=task,
                components=tuple(components),
                expression=Combination("sum", tuple(range(len(components)))),
                calibration_rows=1024,
            )
            gp_rows = {"small": 64, "medium": 512, "large": 1024}[size]
            recipes["gp"] = GaussianProcessRecipe(
                task=replace(task, rows=gp_rows, context_rows=gp_rows * 3 // 4),
                calibration_rows=512,
            )
            cases.extend(
                {"name": f"matrix-{family}-{size}-{kind}", "recipe": asdict(recipe)}
                for family, recipe in recipes.items()
            )
    return cases


def write(path, value):
    write_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def write_bytes(path, payload):
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix(".pending")
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def compare(recipe, seed, task_id, candidate):
    import torch

    from r0.modules.priors import generate_task

    reference = generate_task(recipe, seed=seed, task_id=task_id, device="cpu")
    other = generate_task(recipe, seed=seed, task_id=task_id, device=candidate)
    for field in ("source_task_id", "episode_id", "mechanism_json", "classes"):
        if getattr(reference, field) != getattr(other, field):
            raise ValueError(f"different {field}")
    errors = {}
    for field in (
        "inputs",
        "scores",
        "targets",
        "context_indices",
        "query_indices",
        "probabilities",
    ):
        left, right = getattr(reference, field), getattr(other, field)
        if left is None:
            if right is not None:
                raise ValueError(f"unexpected candidate {field}")
            continue
        if right is None:
            raise ValueError(f"missing candidate {field}")
        if left.device.type != "cpu" or right.device.type != candidate:
            raise ValueError(f"unexpected tensor device for {field}")
        right = right.cpu()
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)
        errors[field] = float((left.to(torch.float64) - right.to(torch.float64)).abs().max())
    return {
        "source_task_id": reference.source_task_id,
        "episode_id": reference.episode_id,
        "rows": reference.inputs.shape[0],
        "features": reference.inputs.shape[1],
        "task_type": reference.task_type,
        "classes": len(reference.classes),
        "max_absolute_error": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    override = os.environ.get("R0_GENERATION_SOURCE")
    if (root / ".git").exists() or not override:
        source = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if override and override != source:
            raise ValueError("source override differs from checkout")
    else:
        source = override
    plan_bytes = (
        args.plan.read_bytes()
        if args.plan
        else (
            json.dumps(
                {
                    "source": source,
                    "agreement_suite": "fixed-device-agreement-v1",
                    "request": {"seed": 27, "start": 0},
                    "agreement_cases": agreement_matrix(),
                },
                allow_nan=False,
            ).encode()
        )
    )
    plan = json.loads(plan_bytes)
    if plan["source"] != source:
        raise ValueError("comparison plan belongs to a different source revision")
    cases = plan["agreement_cases"]
    names = [case["name"] for case in cases]
    if not 1 <= len(cases) <= 128 or len(set(names)) != len(names):
        raise ValueError("provide 1..128 uniquely named comparison cases")
    args.out.mkdir(parents=True, exist_ok=False)
    write_bytes(args.out / "plan.json", plan_bytes)
    if args.plan_only:
        write(
            args.out / "summary.json",
            {
                "status": "PLAN_ONLY",
                "planned": len(cases),
                "passed": 0,
                "failed": 0,
                "not_run": len(cases),
            },
        )
        return 0
    write(args.out / "started.json", {"pid": os.getpid(), "candidate_device": args.candidate})

    import numpy as np
    import torch

    from r0.modules.priors import recipe_from_dict

    cuda_available = torch.cuda.is_available() if args.candidate == "cuda" else False
    manifest = {
        "source": source,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "reference_device": "cpu",
        "candidate_device": args.candidate,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if cuda_available else None,
        "cpu_threads": torch.get_num_threads(),
        "rtol": 1e-10,
        "atol": 1e-10,
    }
    write(args.out / "manifest.json", manifest)
    outcomes = []
    status = "PASS"
    if args.candidate == "cuda" and not cuda_available:
        status = "CUDA_UNAVAILABLE"
    else:
        for index, case in enumerate(cases):
            write(args.out / f"case-{index}-started.json", {"name": case["name"]})
            started = time.perf_counter()
            record = {"name": case["name"], "status": "PASS"}
            try:
                record.update(
                    compare(
                        recipe_from_dict(case["recipe"]),
                        plan["request"]["seed"],
                        plan["request"]["start"],
                        args.candidate,
                    )
                )
            except (ValueError, TypeError, KeyError, RuntimeError, AssertionError) as error:
                record.update(status="FAILED", error=f"{type(error).__name__}: {error}"[:2048])
                status = "FAILED"
            record["elapsed_seconds"] = time.perf_counter() - started
            write(args.out / f"case-{index}.json", record)
            outcomes.append(record)
            print(f"{index + 1}/{len(cases)} {case['name']}: {record['status']}", flush=True)
            if status != "PASS":
                break
    summary = {
        "status": status,
        "comparison": f"cpu_to_{args.candidate}",
        "planned": len(cases),
        "passed": sum(record["status"] == "PASS" for record in outcomes),
        "failed": sum(record["status"] == "FAILED" for record in outcomes),
        "not_run": len(cases) - len(outcomes),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda_available else None,
    }
    write(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
