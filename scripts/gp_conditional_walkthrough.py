"""Retain bounded conditional GP values, numerical limits and additional-draw checks."""

import argparse
import json
import platform
import resource
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r0.modules.priors import (
    GaussianProcessRecipe,
    GibbsKernel,
    LinearRecipe,
    RationalQuadraticKernel,
    SpectralKernel,
    StationaryKernel,
    condition_gp,
    generate_task,
    record_fingerprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    kernels = (
        StationaryKernel(),
        StationaryKernel(kind="matern12"),
        StationaryKernel(kind="matern32"),
        StationaryKernel(kind="matern52"),
        RationalQuadraticKernel(alpha=0.5),
        SpectralKernel(frequencies=(0.1, 0.2, 0.3, 0.4)),
        GibbsKernel(length_slopes=(0.2, -0.2, 0.3, 0.1), amplitude_slopes=(0.2, 0, 0, 0)),
    )
    base = GaussianProcessRecipe(task=LinearRecipe(rows=16, context_rows=8), calibration_rows=32)
    variants = [(k.kind, replace(base, kernels=(k,))) for k in kernels]
    variants.extend(
        (mode, replace(base, kernels=kernels, combine=mode)) for mode in ("sum", "product")
    )
    cases = []
    for name, recipe in variants:
        for task_type, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
            cases.append(
                (
                    f"{name}: {task_type}",
                    replace(
                        recipe, task=replace(recipe.task, task_type=task_type, classes=classes)
                    ),
                    3,
                )
            )
    cases.append(
        (
            "maximum dimensions",
            replace(
                base,
                task=LinearRecipe(
                    rows=1024,
                    context_rows=512,
                    features=256,
                    signal_features=256,
                    active_features=4,
                    task_type="multiclass",
                    classes=100,
                ),
                calibration_rows=512,
                kernels=(StationaryKernel(kind="matern12"),),
            ),
            256,
        )
    )
    records: list[dict[str, Any]] = []
    for i, (name, recipe, count) in enumerate(cases):
        task = generate_task(recipe, seed=27, task_id=i)
        before = record_fingerprint(
            (asdict(task.episode()), task.mechanism_json, task.scores.tolist())
        )
        query = task.inputs[:count].clone() * 0.7 + 0.1
        started = time.perf_counter()
        result = condition_gp(task, query, draws=3, seed=42)
        elapsed = time.perf_counter() - started
        saved = condition_gp(task, task.inputs[:2], draws=2, seed=42)
        error = float(np.max(np.abs(np.asarray(saved["draws"])[0] - task.scores[:2].numpy())))
        assert error < 1e-9
        assert before == record_fingerprint(
            (asdict(task.episode()), task.mechanism_json, task.scores.tolist())
        )
        assert result == condition_gp(task, query, draws=3, seed=42)
        assert np.isfinite(result["draws"]).all()
        (args.out / f"case-{i}.json").write_text(json.dumps(result, allow_nan=False))
        records.append(
            {
                "case": i,
                "name": name,
                "identity": task.episode_id,
                "query_count": count,
                "seconds": elapsed,
                "saved_max_error": error,
                "data_fingerprint": before,
                "conditioning": result["conditioning"],
            }
        )
        print(name, "PASS", flush=True)
    # Independent dense solve and empirical covariance from 1024 additional draws.
    task = generate_task(
        replace(base, kernels=(StationaryKernel(kind="matern12"),)), seed=27, task_id=99
    )
    card = json.loads(task.mechanism_json)
    query = task.inputs[:3] * 0.7 + 0.1
    support = np.array(card["support"])
    points = query.numpy()[:, card["active"]]

    def matern(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-np.sqrt(((a[:, None] - b[None, :]) ** 2).sum(2)))

    k, cross = matern(support, support), matern(points, support)
    eigenvalues, vectors = np.linalg.eigh(k)
    raw = (
        (vectors * np.sqrt(np.maximum(eigenvalues, 0)))
        @ vectors.T
        @ np.array(card["standard_normal"])
    )
    expected_mean = ((cross @ np.linalg.solve(k, raw) - card["center"]) / card["scale"])[:, 0]
    expected_covariance = (matern(points, points) - cross @ np.linalg.solve(k, cross.T)) / card[
        "scale"
    ][0] ** 2
    samples = np.concatenate(
        [
            np.asarray(condition_gp(task, query, draws=8, seed=seed)["draws"])[:, :, 0]
            for seed in range(128)
        ]
    )
    standard_errors = np.sqrt(np.diag(expected_covariance) / len(samples))
    mean_z = np.abs(samples.mean(0) - expected_mean) / standard_errors
    covariance_error = np.max(np.abs(np.cov(samples, rowvar=False) - expected_covariance))
    assert np.max(mean_z) < 5
    assert covariance_error < 0.15 * np.max(np.diag(expected_covariance))
    np.savez(
        args.out / "additional-draws.npz",
        samples=samples,
        expected_mean=expected_mean,
        expected_covariance=expected_covariance,
    )
    rank_task = generate_task(
        replace(
            base,
            task=LinearRecipe(
                rows=16, context_rows=8, features=1, signal_features=1, active_features=1
            ),
            kernels=(StationaryKernel(length_scale=2),),
        ),
        seed=27,
        task_id=8,
    )
    rank_query = torch.tensor([[-0.5], [0.1], [0.7]], dtype=torch.float64)
    try:
        condition_gp(rank_task, rank_query)
    except ValueError as exc:
        rank_failure = str(exc)
    else:
        raise AssertionError("rank deficient support must fail strict inspection")
    approximate = condition_gp(rank_task, rank_query, draws=2, seed=42, allow_rank_truncation=True)
    (args.out / "approximate.json").write_text(json.dumps(approximate, indent=2, allow_nan=False))
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "scope": "local CPU API; no UI, hosted or predictive performance evidence",
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cases": records,
                "sampling": {
                    "draws": len(samples),
                    "maximum_mean_standard_errors": float(np.max(mean_z)),
                    "covariance_max_error": float(covariance_error),
                    "expected_max_variance": float(np.max(np.diag(expected_covariance))),
                },
                "strict_rank_failure": rank_failure,
                "approximate_conditioning": approximate["conditioning"],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
