"""Bounded local evidence: distributions, scalar truth and measured saved tasks."""

import argparse
import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

from r0.modules.generation import RunRequest, run_guarded
from r0.modules.priors import InputDistribution, LinearRecipe, PolynomialRecipe, generate_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    cases = [(name, 4, None, 128, 8) for name in ("gaussian", "uniform", "student_t", "mixture")]
    cases += [("student_t", 16, 2.0, 128, 8), ("gaussian", 3, None, 20000, 256)]
    for i, (name, degree, bound, rows, columns) in enumerate(cases):
        recipe = PolynomialRecipe(
            task=LinearRecipe(
                rows=rows,
                context_rows=rows // 2,
                features=columns,
                signal_features=columns,
                active_features=columns,
            ),
            inputs=InputDistribution(distribution=name, df=9, bound=bound),
            degree=degree,
        )
        task = generate_task(recipe, seed=27, task_id=0)
        card = json.loads(task.mechanism_json)
        raw = math.fsum(
            float(task.inputs[0, j]) ** power * weight[0]
            for j, (power, weight) in enumerate(zip(card["powers"], card["weights"], strict=True))
        )
        expected = (raw - card["center"][0]) / card["scale"][0]
        actual = float(task.scores[0, 0])
        assert math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)
        request = RunRequest(
            recipe, seed=27, count=1, group_size=1, max_seconds=60, max_bytes=64 * 1024**2
        )
        result = run_guarded(request, args.out / str(i))
        assert result.returncode == 0, result.stderr
        records.append(
            {
                "request": asdict(request),
                "scalar_expected": expected,
                "scalar_actual": actual,
                "result": asdict(result),
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
