"""Measure smooth functions and reconstruct scalar truth from saved task arrays."""

import argparse
import io
import json
import math
import subprocess
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from r0.modules.generation import RunRequest, run_guarded
from r0.modules.priors import InputDistribution, LinearRecipe, SmoothRecipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [
        SmoothRecipe(family="sinusoidal", projection="additive", width_spread=3),
        SmoothRecipe(
            family="radial",
            projection="additive",
            width_spread=3,
            inputs=InputDistribution(distribution="uniform"),
            task=LinearRecipe(task_type="binary", classes=2),
        ),
        SmoothRecipe(
            family="sinusoidal",
            projection="ridge",
            width_spread=3,
            inputs=InputDistribution(distribution="student_t", df=1),
            task=LinearRecipe(task_type="multiclass", classes=3),
        ),
        SmoothRecipe(
            family="radial",
            projection="ridge",
            width_spread=3,
            inputs=InputDistribution(distribution="mixture"),
        ),
        SmoothRecipe(
            terms=1,
            task=LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
        ),
        SmoothRecipe(
            family="sinusoidal",
            projection="ridge",
            terms=256,
            width_spread=10,
            task=LinearRecipe(
                rows=2048,
                context_rows=1024,
                features=64,
                signal_features=64,
                active_features=32,
                noise_ratio=1,
            ),
        ),
        SmoothRecipe(
            family="radial",
            projection="ridge",
            terms=4096,
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
        SmoothRecipe(
            family="sinusoidal",
            projection="additive",
            terms=4096,
            task=LinearRecipe(
                rows=20000,
                context_rows=10000,
                features=256,
                signal_features=256,
                active_features=256,
            ),
        ),
    ]
    records = []
    for i, recipe in enumerate(cases):
        request = RunRequest(
            recipe, seed=27, count=1, group_size=1, max_seconds=120, max_bytes=256 * 1024**2
        )
        result = run_guarded(request, args.out / str(i))
        assert result.returncode == 0, result.stderr
        assert result.summary is not None
        assert result.summary["completed"] == 1
        archive = next((args.out / str(i)).glob("groups/*/*/data-*.zip"))
        with zipfile.ZipFile(archive) as saved:
            card = json.loads(saved.read("0/card.json"))["mechanism"]
            row = json.loads(saved.read("0/predictor.json"))["context_rows"][0]
            with np.load(io.BytesIO(saved.read("0/predictor.npz")), allow_pickle=False) as data:
                x = [float(v) for v in data["context_x"][0]]
            with np.load(io.BytesIO(saved.read("0/truth.npz")), allow_pickle=False) as data:
                actual = float(data["scores"][row, 0])
        values: list[float] = []
        for direction, center, width in zip(
            card["directions"], card["centers"], card["widths"], strict=True
        ):
            z = (
                math.fsum(
                    v * a for v, a in zip(x[: recipe.task.signal_features], direction, strict=True)
                )
                - center
            ) / width
            values.append(math.sin(z) if recipe.family == "sinusoidal" else math.exp(-z * z / 2))
        raw = math.fsum(v * w[0] for v, w in zip(values, card["weights"], strict=True))
        expected = (raw - card["center"][0]) / card["scale"][0]
        assert math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-10)
        resumed = run_guarded(request, args.out / str(i))
        assert resumed.returncode == 0, resumed.stderr
        assert resumed.summary is not None
        assert resumed.summary["generated"] == 0
        records.append(
            {
                "request": asdict(request),
                "row_index": row,
                "scalar_expected": expected,
                "scalar_actual": actual,
                "receipt": json.loads(archive.with_suffix(".json").read_text()),
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
