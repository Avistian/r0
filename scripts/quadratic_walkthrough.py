"""Retain quadratic scalar reconstruction, boundary shapes and guarded-run costs."""

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import RunRequest, run_guarded
from r0.modules.priors import InputDistribution, LinearRecipe, QuadraticRecipe, generate_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [
        QuadraticRecipe(inputs=InputDistribution(distribution=name, df=5))
        for name in ("gaussian", "uniform", "student_t", "mixture")
    ]
    cases += [
        QuadraticRecipe(
            task=LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
            rank=1,
        ),
        QuadraticRecipe(
            task=LinearRecipe(
                rows=20000,
                context_rows=10000,
                features=256,
                signal_features=256,
                active_features=256,
                task_type="multiclass",
                classes=100,
            ),
            rank=256,
        ),
    ]
    records = []
    for i, recipe in enumerate(cases):
        task = generate_task(recipe, seed=27, task_id=0)
        card = json.loads(task.mechanism_json)
        x = [float(v) for v in task.inputs[0]]
        # Expand the quadratic matrix rather than using the factorized generator expression.
        raw = math.fsum(
            x[j]
            * x[k]
            * math.fsum(
                w[0] * a[j] * a[k] for a, w in zip(card["directions"], card["weights"], strict=True)
            )
            * (1 if j == k else recipe.interaction_strength)
            for j in range(recipe.task.signal_features)
            for k in range(recipe.task.signal_features)
        ) + recipe.linear_strength * math.fsum(
            x[j] * w[0] for j, w in enumerate(card["linear_weights"])
        )
        expected = (raw - card["center"][0]) / card["scale"][0]
        actual = float(task.scores[0, 0])
        assert math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)
        request = RunRequest(
            recipe, seed=27, count=1, group_size=1, max_seconds=60, max_bytes=128 * 1024**2
        )
        result = run_guarded(request, args.out / str(i))
        assert result.returncode == 0, result.stderr
        assert result.summary is not None
        assert result.summary["completed"] == 1
        resumed = run_guarded(request, args.out / str(i))
        assert resumed.returncode == 0, resumed.stderr
        assert resumed.summary is not None
        assert resumed.summary["generated"] == 0
        records.append(
            {
                "request": asdict(request),
                "scalar_expected": expected,
                "scalar_actual": actual,
                "result": asdict(result),
                "resume": asdict(resumed),
            }
        )
    print(
        json.dumps(
            {
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "cases": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
