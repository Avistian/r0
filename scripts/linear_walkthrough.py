"""Retained independent arithmetic and size evidence for the first #27 slice."""

import argparse
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path

import torch

from r0.modules.priors import LinearRecipe, generate_linear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    recipe = LinearRecipe(features=4, signal_features=4, active_features=2)
    task = generate_linear(recipe, seed=11, task_id=2, device=args.device)
    mechanism = json.loads(task.mechanism_json)
    row = task.inputs[0].cpu().tolist()
    weights = [w[0] for w in mechanism["weights"]]
    calculated = math.fsum(x * w for x, w in zip(row, weights, strict=True))
    actual = float(task.scores[0, 0])
    assert math.isclose(calculated, actual, rel_tol=1e-10, abs_tol=1e-12)
    noisy = generate_linear(replace(recipe, noise_ratio=1), seed=11, task_id=2, device=args.device)
    twice = generate_linear(replace(recipe, noise_ratio=2), seed=11, task_id=2, device=args.device)
    torch.testing.assert_close(twice.targets - task.targets, 2 * (noisy.targets - task.targets))
    workloads: list[dict[str, object]] = []
    for rows, features in ((128, 8), (2048, 64), (20_000, 256)):
        current = LinearRecipe(
            rows=rows,
            features=features,
            signal_features=features,
            active_features=features,
            context_rows=rows // 2,
        )
        # Warm-up is separate from the three timed generations.
        generate_linear(current, seed=37, task_id=0, device=args.device)
        timings: list[float] = []
        for i in range(3):
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = generate_linear(current, seed=37, task_id=i + 1, device=args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            assert result.inputs.shape == (rows, features)
        workloads.append({"rows": rows, "features": features, "seconds": timings})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "device": args.device,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cpu_threads": torch.get_num_threads(),
                "recipe": json.loads(task.recipe_json),
                "mechanism": mechanism,
                "worked_row": {
                    "inputs": row,
                    "scalar_calculation": calculated,
                    "generated": actual,
                },
                "noise_pair": "PASS",
                "workloads": workloads,
                "timing_scope": (
                    "generation including canonical CPU sampling and device transfer; "
                    "excludes episode conversion/export"
                ),
                "gpu_agreement": "NOT_RUN"
                if args.device == "cpu"
                else "see numerical device tests",
                "peak_memory": "NOT_MEASURED",
                "hosted_execution": "NOT_RUN",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
