"""Exercise bounded episode batches with a small differentiable context predictor."""

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch

from r0.modules.episodes import Episode, PredictorView
from r0.modules.featurize import transform
from r0.modules.generation import load_group
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
    Recipe,
    SmoothRecipe,
    generate_task,
)


def portfolio() -> list[tuple[str, Recipe]]:
    """A fixed verification sweep; no production mixture or eligibility weights."""
    cases: list[tuple[str, Recipe]] = []
    for i, (kind, classes) in enumerate((("regression", 0), ("binary", 2), ("multiclass", 3))):
        controls = LinearRecipe(
            task_type=kind,
            classes=classes,
            rows=32 + 16 * i,
            context_rows=20 + 8 * i,
            features=8 + 4 * i,
            noise_ratio=0.2 if kind == "regression" else 0,
        )
        child = replace(controls, noise_ratio=0)
        cases.extend(
            [
                ("linear", replace(controls, classes=100) if kind == "multiclass" else controls),
                ("grouped", GroupedLinearRecipe(task=controls, calibration_rows=256)),
                ("polynomial", PolynomialRecipe(task=controls, calibration_rows=256)),
                ("quadratic", QuadraticRecipe(task=controls, calibration_rows=256)),
                ("rff", FourierRecipe(task=controls, terms=64, calibration_rows=256)),
                (
                    "periodic",
                    FourierRecipe(task=controls, family="periodic", terms=8, calibration_rows=256),
                ),
                ("radial", SmoothRecipe(task=controls, calibration_rows=256)),
                (
                    "sinusoidal",
                    SmoothRecipe(task=controls, family="sinusoidal", calibration_rows=256),
                ),
                ("gp", GaussianProcessRecipe(task=controls, calibration_rows=32)),
                (
                    "composition",
                    CompositeRecipe(
                        task=controls,
                        components=(
                            FunctionComponent(child),
                            FunctionComponent(PolynomialRecipe(task=child, calibration_rows=256)),
                        ),
                        expression=Combination("sum", (0, 1)),
                        calibration_rows=256,
                    ),
                ),
            ]
        )
    return cases


def produce(
    args: argparse.Namespace, progress: dict[str, Any]
) -> Iterator[tuple[dict[str, Any], Episode]]:
    cases = portfolio()
    saved = None if args.archive is None else load_group(args.archive)
    for task_id in range(args.count):
        if saved is None:
            name, recipe = cases[task_id % len(cases)]
            progress["attempt"] = {
                "task_id": task_id,
                "seed": args.seed,
                "family": name,
                "recipe": asdict(recipe),
            }
            task = generate_task(recipe, seed=args.seed, task_id=task_id)
        else:
            progress["attempt"] = {"archive_index": task_id}
            try:
                task = next(saved)
            except StopIteration:
                return
            name = "saved"
        record = {
            "source_task_id": task.source_task_id,
            "episode_id": task.episode_id,
            "family": name,
            "recipe": json.loads(task.recipe_json),
        }
        episode = task.episode()
        del task
        progress["produced"] += 1
        yield record, episode
        del record, episode


def predict(view: PredictorView, log_bandwidth: torch.Tensor) -> torch.Tensor:
    """Only inference-available fields enter the predictor; no generator dispatch."""
    if any(feature.kind not in ("continuous", "count") for feature in view.features):
        raise ValueError("this bounded example requires numeric features")
    encoded = transform(view)
    if any(any(row.missing) or any(row.unknown) for row in encoded.context + encoded.queries):
        raise ValueError("this bounded example requires complete numeric features")
    context = torch.tensor([row.values for row in encoded.context], dtype=torch.float64)
    query = torch.tensor([row.values for row in encoded.queries], dtype=torch.float64)
    distances = torch.cdist(query, context).square() / context.shape[1]
    weights = torch.softmax(-distances / log_bandwidth.exp(), dim=1)
    if view.task_type == "regression":
        values = torch.tensor(view.context_targets, dtype=torch.float64)
        return weights @ values
    indices = torch.tensor([view.classes.index(str(value)) for value in view.context_targets])
    indicators = torch.nn.functional.one_hot(indices, num_classes=len(view.classes)).to(
        torch.float64
    )
    # A fixed uniform mass keeps absent context classes legal and the loss finite.
    return 0.99 * (weights @ indicators) + 0.01 / len(view.classes)


def query_loss(episode: Episode, prediction: torch.Tensor) -> torch.Tensor:
    if episode.task_type == "regression":
        truth = torch.tensor(episode.query_targets, dtype=torch.float64)
        return (prediction - truth).square().mean()
    truth = torch.tensor([episode.classes.index(str(value)) for value in episode.query_targets])
    return -prediction[torch.arange(len(truth)), truth].log().mean()


def step(
    batch: tuple[tuple[dict[str, Any], Episode], ...],
    parameter: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    progress: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Accumulate one mean objective and update the shared scalar once per batch."""
    losses = []
    records = []
    predictions = {}
    before = float(parameter.detach())
    optimizer.zero_grad()
    for i, (record, episode) in enumerate(batch):
        progress["attempt"] = record
        rows = len(episode.context) + len(episode.queries)
        if rows > 128 or len(episode.features) > 256 or len(episode.classes) > 100:
            raise ValueError(
                "bounded consumer example accepts at most 128 rows, 256 features and 100 classes"
            )
        prediction = predict(episode.predictor_view(), parameter)
        changed_targets = (
            tuple(float(value) + 1000 for value in episode.query_targets)
            if episode.task_type == "regression"
            else tuple(
                episode.classes[(episode.classes.index(str(value)) + 1) % len(episode.classes)]
                for value in episode.query_targets
            )
        )
        changed = replace(episode, query_targets=changed_targets)
        with torch.no_grad():
            assert torch.equal(prediction.detach(), predict(changed.predictor_view(), parameter))
        loss = query_loss(episode, prediction)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite training objective")
        losses.append(loss)
        predictions[f"task_{i}"] = prediction.detach().numpy()
        records.append(
            {
                **record,
                "task_type": episode.task_type,
                "rows": rows,
                "features": len(episode.features),
                "context_rows": len(episode.context),
                "query_rows": len(episode.queries),
                "classes": len(episode.classes),
                "missing_context_classes": len(
                    set(episode.classes) - set(map(str, episode.context_targets))
                )
                if episode.classes
                else 0,
                "loss": float(loss.detach()),
                "predictions": f"task_{i}",
                "query_target_counterfactual": "PASS",
            }
        )
    objective = torch.stack(losses).mean()
    objective.backward()
    if parameter.grad is None or not bool(torch.isfinite(parameter.grad)):
        raise ValueError("missing or nonfinite gradient")
    gradient = float(parameter.grad)
    optimizer.step()
    if not math.isfinite(float(parameter.detach())):
        raise ValueError("nonfinite updated bandwidth")
    return {
        "parameter_before": before,
        "parameter_after": float(parameter.detach()),
        "gradient": gradient,
        "objective": float(objective.detach()),
        "episodes": records,
    }, predictions


def main(args: argparse.Namespace) -> None:
    if not 1 <= args.count <= 300 or not 1 <= args.batch_size <= 8 or not 0 <= args.seed < 2**63:
        raise ValueError("example limits: count 1..300, batch size 1..8, seed in [0,2**63)")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src"], check=False).returncode:
        raise ValueError("commit source changes before retaining revision-labelled evidence")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    progress: dict[str, Any] = {
        "produced": 0,
        "consumed": 0,
        "batches": 0,
        "max_batch_episodes": 0,
        "max_batch_rows": 0,
        "max_batch_distance_cells": 0,
    }
    run = {
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "requested_episodes": args.count,
        "batch_size": args.batch_size,
        "archive": None if args.archive is None else str(args.archive),
        "archive_sha256": None,
        "learning_rate": 0.05,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "device": "cpu",
        "status": "STARTED",
    }
    if args.archive is not None:
        with args.archive.open("rb") as archive:
            run["archive_sha256"] = hashlib.file_digest(archive, "sha256").hexdigest()
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    source = produce(args, progress)
    try:
        with (args.out / "batches.jsonl").open("x") as stream:
            while True:
                batch = tuple(islice(source, args.batch_size))
                if not batch:
                    break
                assert progress["produced"] - progress["consumed"] == len(batch)
                report, predictions = step(batch, parameter, optimizer, progress)
                progress["max_batch_episodes"] = max(progress["max_batch_episodes"], len(batch))
                progress["max_batch_rows"] = max(
                    progress["max_batch_rows"],
                    sum(len(e.context) + len(e.queries) for _, e in batch),
                )
                progress["max_batch_distance_cells"] = max(
                    progress["max_batch_distance_cells"],
                    sum(len(e.context) * len(e.queries) for _, e in batch),
                )
                progress["consumed"] += len(batch)
                filename = f"batch-{progress['batches']}.npz"
                np.savez(args.out / filename, **predictions)
                stream.write(
                    json.dumps({**report, "prediction_file": filename}, allow_nan=False) + "\n"
                )
                stream.flush()
                progress["batches"] += 1
                del batch, predictions, report
        if not progress["consumed"]:
            raise ValueError("no saved episodes were available")
        summary = {
            **run,
            **progress,
            "status": "COMPLETE",
            "final_log_bandwidth": float(parameter.detach()),
            "elapsed_seconds": time.monotonic() - started,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
        }
        summary.pop("attempt", None)
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(summary))
    except (ValueError, RuntimeError, OSError, AssertionError, KeyboardInterrupt) as error:
        (args.out / "failure.json").write_text(
            json.dumps(
                {**run, **progress, "status": "FAILED", "error": repr(error)},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
