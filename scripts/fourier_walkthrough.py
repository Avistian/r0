"""Bounded CPU evidence for the public finite-function generator."""

import json
import math
import platform
import time
from dataclasses import asdict

import numpy as np
import torch

from r0.modules.priors import FourierRecipe, LinearRecipe, generate_fourier


def main() -> None:
    records = []
    for family, terms, rows, features in (
        ("periodic", 4, 128, 8),
        ("rff", 256, 2048, 64),
        ("rff", 64, 20000, 256),
        ("rff", 4096, 128, 8),
    ):
        recipe = FourierRecipe(
            task=LinearRecipe(
                rows=rows,
                features=features,
                signal_features=features,
                active_features=min(features, terms) if family == "periodic" else features,
                context_rows=rows // 2,
            ),
            family=family,
            terms=terms,
        )
        start = time.perf_counter()
        warmup = generate_fourier(recipe, seed=27, task_id=0)
        warmup_seconds = time.perf_counter() - start
        timings = []
        for task_id in (1, 2, 3):
            start = time.perf_counter()
            generated = generate_fourier(recipe, seed=27, task_id=task_id)
            timings.append(time.perf_counter() - start)
            assert torch.isfinite(generated.targets).all()
        card = json.loads(warmup.mechanism_json)
        row = [float(warmup.inputs[0, j]) for j in range(features)]
        raw = math.fsum(
            math.cos(math.fsum(x * w for x, w in zip(row, frequency, strict=True)) + phase)
            * weight[0]
            for frequency, phase, weight in zip(
                card["frequencies"], card["phases"], card["coefficients"], strict=True
            )
        )
        expected = (raw - card["center"][0]) / card["scale"][0]
        actual = float(warmup.scores[0, 0])
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
        records.append(
            {
                "recipe": asdict(recipe),
                "warmup_seconds": warmup_seconds,
                "generation_seconds": timings,
                "scalar_expected": expected,
                "scalar_actual": actual,
                "memory": "NOT_MEASURED",
            }
        )
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "threads": torch.get_num_threads(),
                "device": "cpu",
                "cases": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
