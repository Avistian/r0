"""Measure grouped linear tasks with retained scalar truth and restart evidence."""

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import RunRequest, run_guarded
from r0.modules.priors import GroupedLinearRecipe, InputDistribution, LinearRecipe, generate_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [
        GroupedLinearRecipe(),
        GroupedLinearRecipe(
            inputs=InputDistribution(distribution="uniform"),
            task=LinearRecipe(task_type="binary", classes=2),
        ),
        GroupedLinearRecipe(
            inputs=InputDistribution(distribution="student_t", df=3),
            task=LinearRecipe(task_type="multiclass", classes=3),
        ),
        GroupedLinearRecipe(inputs=InputDistribution(distribution="mixture")),
        GroupedLinearRecipe(
            group_size=1,
            task=LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
        ),
        GroupedLinearRecipe(
            group_size=8,
            task=LinearRecipe(
                rows=2048,
                context_rows=1024,
                features=64,
                signal_features=64,
                active_features=32,
                noise_ratio=1,
            ),
        ),
        GroupedLinearRecipe(
            group_size=256,
            task=LinearRecipe(
                rows=20000,
                context_rows=10000,
                features=256,
                signal_features=256,
                active_features=256,
                task_type="multiclass",
                classes=100,
            ),
        ),
    ]
    records = []
    for i, recipe in enumerate(cases):
        task = generate_task(recipe, seed=27, task_id=0)
        card = json.loads(task.mechanism_json)
        active = set(card["active"])
        assert len(active) == recipe.task.active_features
        for group in card["groups"]:
            assert len(active.intersection(group)) in (0, recipe.group_size)
        raw = math.fsum(float(task.inputs[0, j]) * w[0] for j, w in enumerate(card["weights"]))
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
