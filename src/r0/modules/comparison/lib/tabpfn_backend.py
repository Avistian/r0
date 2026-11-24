"""Local TabPFN loading backend, invoked only by an explicitly provisioned worker."""

import importlib
import os
import zipfile
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Array
from .storage import Document, file_hash
from .tabpfn import tabpfn_recipe


def validate_tabpfn_assets(wheel: Path, model: Path, task: str) -> None:
    recipe = tabpfn_recipe(task)
    if not wheel.is_file() or file_hash(wheel) != recipe["wheel_sha256"]:
        raise ValueError("TabPFN wheel hash mismatch")
    if not model.is_file() or file_hash(model) != recipe["weight_sha256"]:
        raise ValueError("TabPFN weight hash mismatch")
    installed = distribution("tabpfn")
    if installed.version != "2.2.1":
        raise ValueError("installed TabPFN version mismatch")
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith("tabpfn/") and not name.endswith("/"):
                path = Path(str(installed.locate_file(name)))
                if not path.is_file() or path.read_bytes() != archive.read(name):
                    raise ValueError(f"installed TabPFN differs from pinned wheel: {name}")


class TabPFNBackend:
    """Fit preprocessing once on context; preserve declared class-column order."""

    def __init__(self, wheel: Path, model: Path) -> None:
        self.wheel = wheel
        self.model_path = model
        self.model: Any = None
        self.context: Document = {}

    def _frame(self, values: list[list[Any]]) -> Any:
        pandas = importlib.import_module("pandas")
        return pandas.DataFrame(
            [[np.nan if v is None else v for v in row] for row in values],
            columns=[feature["name"] for feature in self.context["features"]],
        )

    def fit(self, context: Document) -> None:
        task = context["task"]
        validate_tabpfn_assets(self.wheel, self.model_path, task)
        # Set before imports; the worker also starts in a clean, input-only cwd.
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
        torch = importlib.import_module("torch")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        module = importlib.import_module("tabpfn")
        recipe = tabpfn_recipe(task)
        parameters = dict(recipe["parameters"])
        parameters["inference_precision"] = torch.float32
        parameters["model_path"] = str(self.model_path.resolve())
        parameters["categorical_features_indices"] = [
            i for i, f in enumerate(context["features"]) if f["kind"] == "categorical"
        ]
        parameters["inference_config"] = recipe["inference_config"]
        constructor = module.TabPFNRegressor if task == "regression" else module.TabPFNClassifier
        self.context = context
        self.model = constructor(**parameters)
        targets = (
            context["targets"]
            if task == "regression"
            else [context["classes"].index(t) for t in context["targets"]]
        )
        self.model.fit(self._frame(context["values"]), np.asarray(targets))

    def predict(self, queries: Document) -> tuple[list[str], Array]:
        if self.model is None:
            raise ValueError("backend must be fitted first")
        frame = self._frame(queries["values"])
        if self.context["task"] == "regression":
            values = self.model.predict(frame, output_type="mean")
        else:
            values = self.model.predict_proba(frame)
            order = [
                list(self.model.classes_).index(i) for i in range(len(self.context["classes"]))
            ]
            values = values[:, order]
        return queries["row_ids"], np.asarray(values, dtype=np.float64)


def tabpfn_worker(input_path: Path, output: Path, config_path: Path) -> None:
    """Execute only the pinned recipe; provisioned config cannot relabel settings."""
    from .prediction import prediction_worker
    from .storage import read_json

    view = read_json(input_path)
    config = read_json(config_path)
    recipe = tabpfn_recipe(view["task"])
    identity = config["identity"]
    if identity["settings"] != recipe or identity["checkpoint"] != recipe["weight_sha256"]:
        raise ValueError("TabPFN identity differs from the frozen recipe or weights")
    prediction_worker(
        input_path, output, TabPFNBackend(Path(config["wheel"]), Path(config["model"])), identity
    )
