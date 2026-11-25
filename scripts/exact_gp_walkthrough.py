"""Bounded direct-GP evidence with an independent covariance/target calculation."""

import argparse
import io
import json
import math
import subprocess
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from r0.modules.generation import RunRequest, run_guarded
from r0.modules.priors import (
    GaussianProcessRecipe,
    InputDistribution,
    LinearRecipe,
    StationaryKernel,
)


def scalar_from_card(card: dict[str, Any], row: int) -> float:
    support = np.asarray(card["support"], dtype=np.float64)
    covariance = (
        np.zeros((len(support), len(support)))
        if card["combine"] == "sum"
        else np.ones((len(support), len(support)))
    )
    for kernel in card["kernels"]:
        # Independent squared-distance identity rather than the sampler's hypot accumulation.
        x = support[:, kernel["axes"]] / np.asarray(kernel["length_scales"])
        norm = np.sum(x * x, axis=1)
        squared = np.maximum(norm[:, None] + norm[None, :] - 2 * x @ x.T, 0)
        np.fill_diagonal(squared, 0)
        distance = np.sqrt(squared)
        if kernel["kind"] == "rbf":
            values = np.exp(-squared / 2)
        elif kernel["kind"] == "matern12":
            values = np.exp(-distance)
        elif kernel["kind"] == "matern32":
            t = math.sqrt(3) * distance
            values = (1 + t) * np.exp(-t)
        else:
            t = math.sqrt(5) * distance
            values = (1 + t + t * t / 3) * np.exp(-t)
        if card["combine"] == "sum":
            covariance += kernel["weight"] * values
        else:
            covariance *= values
    eigenvalues, vectors = np.linalg.eigh(covariance)
    factor = (vectors * np.sqrt(np.maximum(eigenvalues, 0))) @ vectors.T
    index = card["task_support_indices"][row]
    raw = math.fsum(
        float(v) * z[0] for v, z in zip(factor[index], card["standard_normal"], strict=True)
    )
    return (raw - card["center"][0]) / card["scale"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [
        GaussianProcessRecipe(kernels=(StationaryKernel(kind=kind),), calibration_rows=64)
        for kind in ("rbf", "matern12", "matern32", "matern52")
    ]
    cases[1] = replace(
        cases[1],
        inputs=InputDistribution(distribution="uniform"),
        task=LinearRecipe(task_type="binary", classes=2),
    )
    cases[2] = replace(
        cases[2],
        inputs=InputDistribution(distribution="student_t", df=1),
        task=LinearRecipe(task_type="multiclass", classes=3),
    )
    cases[3] = replace(cases[3], inputs=InputDistribution(distribution="mixture"))
    subset = (
        StationaryKernel(axes=(0, 1), weight=3),
        StationaryKernel(kind="matern52", axes=(2, 3), feature_scales=(0.5, 2), weight=7),
    )
    cases += [
        GaussianProcessRecipe(task=LinearRecipe(rows=256, context_rows=128), kernels=subset),
        GaussianProcessRecipe(
            task=LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
            calibration_rows=32,
            kernels=(StationaryKernel(kind="matern12"),),
        ),
        GaussianProcessRecipe(
            task=LinearRecipe(rows=16, context_rows=8),
            calibration_rows=32,
            kernels=tuple(StationaryKernel(length_scale=0.5 + i / 64) for i in range(64)),
        ),
        GaussianProcessRecipe(
            task=LinearRecipe(
                rows=1024,
                context_rows=512,
                features=256,
                signal_features=256,
                active_features=256,
                task_type="multiclass",
                classes=100,
            ),
            calibration_rows=512,
            combine="product",
            kernels=(
                StationaryKernel(length_scale=16),
                StationaryKernel(kind="matern52", length_scale=12),
            ),
        ),
    ]
    records = []
    for i, recipe in enumerate(cases):
        request = RunRequest(
            recipe, seed=27, count=1, group_size=1, max_seconds=120, max_bytes=128 * 1024**2
        )
        result = run_guarded(request, args.out / str(i))
        assert result.returncode == 0, result.stderr
        assert result.summary is not None
        assert result.summary["completed"] == 1
        archive = next((args.out / str(i)).glob("groups/*/*/data-*.zip"))
        with zipfile.ZipFile(archive) as saved:
            card = json.loads(saved.read("0/card.json"))["mechanism"]
            row = json.loads(saved.read("0/predictor.json"))["context_rows"][0]
            with np.load(io.BytesIO(saved.read("0/truth.npz")), allow_pickle=False) as data:
                actual = float(data["scores"][row, 0])
        expected = scalar_from_card(card, row)
        assert math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-8), (i, expected, actual)
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
                "roundoff_tolerance": card["roundoff_tolerance"],
                "factor_covariance_max_error": card["factor_covariance_max_error"],
            }
        )
    failed_request = RunRequest(
        GaussianProcessRecipe(kernels=(StationaryKernel(length_scale=1e200),)),
        seed=27,
        count=1,
        group_size=1,
        max_seconds=30,
        max_bytes=1048576,
    )
    failed = run_guarded(failed_request, args.out / "failure")
    assert failed.summary is not None
    assert failed.summary["failed"] == 1
    assert failed.summary["completed"] == 0
    failures = [
        json.loads(line)
        for path in (args.out / "failure").rglob("failures-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    print(
        json.dumps(
            {
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "cases": records,
                "failure": {
                    "request": asdict(failed_request),
                    "result": asdict(failed),
                    "records": failures,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
