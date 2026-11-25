"""One active dataset, generated with the shared bounded runner and saved on disk."""

import base64
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from r0.modules.explorer.lib.exports import replay_bundle
from r0.modules.explorer.lib.inspection import inspect_rows, inspect_task
from r0.modules.generation import RunRequest, load_group, run_guarded
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
    TrainingTask,
    recipe_from_dict,
    record_fingerprint,
)


def _retain_attempt(path: Path, record: dict[str, Any]) -> None:
    pending = path.with_suffix(".pending")
    with pending.open("x") as stream:
        json.dump(record, stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def _identity_numbers(value: Any, *, for_browser: bool) -> Any:
    """Keep 63-bit identities exact across JavaScript's 53-bit integer boundary."""
    if isinstance(value, list):
        return [_identity_numbers(item, for_browser=for_browser) for item in cast(list[Any], value)]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in cast(dict[str, Any], value).items():
        if key in ("seed", "task_id", "split"):
            if for_browser and type(item) is int and item > 2**53 - 1:
                item = str(item)
            elif not for_browser and isinstance(item, str) and item.isascii() and item.isdecimal():
                item = int(item)
        result[key] = _identity_numbers(item, for_browser=for_browser)
    return result


def recipe_examples() -> dict[str, Any]:
    """Editable starting points; not a production mixture or throughput claim."""
    recipes = {
        "Linear": LinearRecipe(),
        "Grouped linear": GroupedLinearRecipe(),
        "Polynomial": PolynomialRecipe(),
        "Quadratic": QuadraticRecipe(),
        "Random Fourier features": FourierRecipe(),
        "Periodic": FourierRecipe(family="periodic", terms=16),
        "Smooth radial": SmoothRecipe(),
        "Smooth sinusoidal": SmoothRecipe(family="sinusoidal"),
        "Exact Gaussian process": GaussianProcessRecipe(),
        "Composition": CompositeRecipe(
            components=(FunctionComponent(LinearRecipe()), FunctionComponent(PolynomialRecipe())),
            expression=Combination("product", (0, 1)),
        ),
    }
    return {name: asdict(recipe) for name, recipe in recipes.items()}


class ExplorerSession:
    """Shared actions for local browser and embedded notebook transports."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.task: TrainingTask | None = None
        self.archive: str | None = None
        self.run: dict[str, Any] | None = None

    def _view(self) -> dict[str, Any]:
        view = None if self.task is None else inspect_task(self.task)
        if view is not None:
            view["recipe"] = _identity_numbers(view["recipe"], for_browser=True)
        return {
            "view": view,
            "recipe_json": None if self.task is None else self.task.recipe_json,
            "archive": self.archive,
            "run": self.run,
        }

    def _open(self, path: Path, task_id: int) -> None:
        path = path.resolve()
        if not path.is_relative_to(self.output_dir):
            raise ValueError("archive must be inside the explorer output directory")
        for task in load_group(path):
            if json.loads(task.recipe_json)["task_id"] == task_id:
                self.task = task
                self.archive = str(path.relative_to(self.output_dir))
                return
        raise ValueError("task identifier was not found in this archive")

    def request(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        if action == "examples":
            return {"examples": recipe_examples(), "output_dir": str(self.output_dir)}
        if action == "view":
            return self._view()
        if action == "open":
            self._open(
                self.output_dir / data["archive"],
                _identity_numbers(data, for_browser=False)["task_id"],
            )
            self.run = None
            return self._view()
        if action == "generate":
            record = (
                json.loads(data["record_json"])
                if "record_json" in data
                else data.get("record", data)
            )
            if not isinstance(record, dict):
                raise ValueError("recipe record must be an object")
            record = _identity_numbers(record, for_browser=False)
            device = data.get("device", record.get("execution", {}).get("device", "cpu"))
            request = RunRequest(
                recipe=recipe_from_dict(record["recipe"]),
                seed=record["seed"],
                start=record["task_id"],
                count=1,
                group_size=1,
                max_seconds=data.get("max_seconds", 60),
                max_bytes=data.get("max_bytes", 256 * 1024**2),
                device=device,
            )
            out = self.output_dir / record_fingerprint(asdict(request))
            attempts = self.output_dir / ".attempts" / out.name
            attempts.mkdir(parents=True, exist_ok=True)
            attempt = uuid.uuid4().hex
            record = {"attempt": attempt, "request": asdict(request), "output": str(out)}
            _retain_attempt(attempts / f"{attempt}-started.json", {**record, "status": "STARTED"})
            try:
                run = run_guarded(request, out)
            except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
                _retain_attempt(
                    attempts / f"{attempt}-failed.json",
                    {**record, "status": "FAILED", "error": repr(error)},
                )
                raise
            failed = run.timed_out or bool(run.returncode) or run.summary is None
            _retain_attempt(
                attempts / f"{attempt}-result.json",
                {**record, "status": "FAILED" if failed else "COMPLETE", "result": asdict(run)},
            )
            if failed:
                raise ValueError(
                    f"Generation failed; attempt records retained in {attempts}. "
                    f"Timed out: {run.timed_out}. {run.stderr[-2000:]} {run.summary}"
                )
            archives = sorted(out.glob("groups/**/*.zip"))
            if len(archives) != 1:
                raise ValueError(f"Expected one completed task archive in {out}")
            self._open(archives[0], request.start)
            self.run = asdict(run)
            return self._view()
        if self.task is None:
            raise ValueError("generate or open a dataset first")
        if action == "row":
            return {"row": inspect_rows(self.task, [data["index"]])[0]}
        if action == "parameters":
            return {"parameters": json.loads(self.task.mechanism_json)}
        if action == "export":
            bundle = replay_bundle(self.task, max_bytes=data.get("max_bytes", 256 * 1024**2))
            return {
                "name": f"dataset-{self.task.source_task_id}.zip",
                "data": base64.b64encode(bundle).decode("ascii"),
                "mime": "application/zip",
            }
        raise ValueError(f"unknown explorer action: {action}")
