"""One active dataset, generated with the shared bounded runner and saved on disk."""

import base64
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from r0.modules.explorer.lib.comparisons import SavedPair, create_pair, open_pair, pair_bundle
from r0.modules.explorer.lib.construction import (
    caller_limits,
    construction_examples,
    frozen_archive_version,
    guarded_archive_version,
    population_identity,
    prepare_mechanism_editor,
)
from r0.modules.explorer.lib.curves import inspect_gp_curve, inspect_response_curve
from r0.modules.explorer.lib.edits import SavedEdit, create_edit, open_edit
from r0.modules.explorer.lib.exports import replay_bundle
from r0.modules.explorer.lib.inspection import inspect_rows, inspect_task
from r0.modules.explorer.lib.interventions import (
    guarded_intervention,
    intervention_controls,
    intervention_download,
    saved_intervention,
)
from r0.modules.explorer.lib.saved import load_saved, run_saved, run_saved_document
from r0.modules.generation import RecipeSampling, RunRequest
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
    sample_recipe,
)


def _identity_numbers(value: Any, *, for_browser: bool) -> Any:
    """Keep 63-bit identities exact across JavaScript's 53-bit integer boundary."""
    if isinstance(value, list):
        return [_identity_numbers(item, for_browser=for_browser) for item in cast(list[Any], value)]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in cast(dict[str, Any], value).items():
        if key in ("seed", "task_id", "split", "population", "reference_seed", "reference_task_id"):
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
        self.pair: SavedPair | None = None
        self.edit: SavedEdit | None = None
        self.intervention: SavedPair | None = None

    def _view(self) -> dict[str, Any]:
        view = None if self.task is None else inspect_task(self.task)
        if view is not None:
            view["recipe"] = _identity_numbers(view["recipe"], for_browser=True)
        return {
            "view": view,
            "recipe_json": None if self.task is None else self.task.recipe_json,
            "edit": None
            if self.edit is None
            else {
                "path": self.edit.path,
                "card": self.edit.document["card"],
                "after": _identity_numbers(inspect_task(self.edit.after), for_browser=True),
                "after_archive": self.edit.document["after_archive"],
            },
            "intervention": None
            if self.intervention is None
            else _identity_numbers(
                {
                    "path": self.intervention.path,
                    "card": self.intervention.document["card"],
                    "before_id": self.intervention.before.episode_id,
                    "after_id": self.intervention.after.episode_id,
                    "after": inspect_task(self.intervention.after),
                },
                for_browser=True,
            ),
            "archive": self.archive,
            "run": self.run,
            "pair": None
            if self.pair is None
            else {
                "path": self.pair.path,
                "card": _identity_numbers(self.pair.document["card"], for_browser=True),
                "card_json": json.dumps(
                    self.pair.document["card"], sort_keys=True, allow_nan=False
                ),
                "run": self.pair.document["after_run"],
                "after": {
                    "view": _identity_numbers(inspect_task(self.pair.after), for_browser=True),
                    "recipe_json": self.pair.after.recipe_json,
                    "archive": self.pair.document["after_archive"],
                },
            },
        }

    def _open(self, path: Path, task_id: int) -> None:
        task = load_saved(self.output_dir, path, task_id)
        self.task = task
        self.archive = str(path.resolve().relative_to(self.output_dir))
        self.pair = None
        self.edit = None
        self.intervention = None

    def request(self, action: str, data: dict[str, Any]) -> dict[str, Any]:
        if action == "construction_examples":
            return {"examples": construction_examples()}
        if action == "construct":
            normalized = _identity_numbers(data, for_browser=False)
            seed = normalized["construction_seed"]
            if isinstance(seed, str) and seed.isdecimal():
                seed = int(seed)
            if type(seed) is not int:
                raise ValueError("construction seed must be an integer")
            request = {
                "construction": {
                    "controls_json": json.dumps(normalized["controls"], allow_nan=False),
                    "seed": seed,
                },
                "seed": normalized["seed"],
                "start": normalized["task_id"],
                "count": 1,
                "group_size": 1,
                "max_seconds": data.get("max_seconds", 60),
                "max_bytes": data.get("max_bytes", 256 * 1024**2),
                "device": data.get("device", "cpu"),
                "causal_limits": asdict(caller_limits(data)),
            }
            self.task, self.archive, self.run = run_saved_document(self.output_dir, request)
            self.pair, self.edit = None, None
            self.intervention = None
            return self._view()
        if action == "open_intervention":
            result = guarded_intervention(self.output_dir, "open", data)
            pair = saved_intervention(self.output_dir, result)
            self.task, self.archive = pair.before, pair.document["before_archive"]
            self.intervention = pair
            self.pair, self.edit, self.run = None, None, None
            return self._view()
        if action == "open_edit":
            edit = open_edit(self.output_dir, data["path"])
            self.task, self.archive, self.edit = edit.before, edit.document["before_archive"], edit
            self.pair, self.run = None, None
            self.intervention = None
            return self._view()
        if action == "examples":
            return {"examples": recipe_examples(), "output_dir": str(self.output_dir)}
        if action == "view":
            return self._view()
        if action == "open_pair":
            pair = open_pair(self.output_dir, data["path"], limits=caller_limits(data))
            self.task, self.archive = pair.before, pair.document["before_archive"]
            self.run, self.pair = pair.document["before_run"], pair
            self.edit = None
            self.intervention = None
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
            record = cast(dict[str, Any], record)
            record = population_identity(record, data)
            if "frozen_identity" in data:
                identity = data["frozen_identity"]
                frozen_recipe = record.get("recipe")
                version = (
                    cast(dict[str, Any], frozen_recipe).get("version", "")
                    if type(frozen_recipe) is dict
                    else ""
                )
                if (
                    "record_json" not in data
                    or type(identity) is not dict
                    or set(cast(dict[str, Any], identity)) != {"seed", "task_id"}
                    or not frozen_archive_version(version)
                ):
                    raise ValueError(
                        "frozen identity changes require an original frozen archive "
                        "and seed/task keys"
                    )
                record = {**record, **cast(dict[str, Any], identity)}
            record = _identity_numbers(record, for_browser=False)
            device = data.get("device", record.get("execution", {}).get("device", "cpu"))
            if (
                "sampling" not in record
                and "sampling" not in data
                and guarded_archive_version(record["recipe"].get("version", ""))
            ):
                request = {
                    "recipe": record["recipe"],
                    "seed": record["seed"],
                    "start": record["task_id"],
                    "count": 1,
                    "group_size": 1,
                    "device": device,
                    "max_seconds": data.get("max_seconds", 60),
                    "max_bytes": data.get("max_bytes", 256 * 1024**2),
                }
                if "causal_limits" in data:
                    request["causal_limits"] = asdict(caller_limits(data))
                self.task, self.archive, self.run = run_saved_document(self.output_dir, request)
                self.pair, self.edit = None, None
                self.intervention = None
                return self._view()
            recipe = recipe_from_dict(record["recipe"], limits=caller_limits(data))
            sampling = None
            if "sampling" in record and "record" not in data and "record_json" not in data:
                # A direct request uses the same top-level fields as a fresh record.
                raise ValueError("put sampled task controls in record and sampling beside it")
            if "sampling" in record:
                if "sampling" in data:
                    raise ValueError("record already contains sampling; do not override it")
                origin = record["sampling"]
                sampling = RecipeSampling(family=origin["family"], profile=origin["profile"])
                controls = recipe_from_dict(origin["task"])
                if not isinstance(controls, LinearRecipe):
                    raise ValueError("sampling requires LinearRecipe task controls")
                selected = sample_recipe(
                    sampling.family,
                    profile=sampling.profile,
                    task=controls,
                    seed=record["seed"],
                    task_id=record["task_id"],
                )
                if selected != recipe:
                    raise ValueError("selected recipe differs from recorded sampling provenance")
                recipe = controls
            elif "sampling" in data:
                sampling = RecipeSampling(**data["sampling"])
            request = RunRequest(
                recipe=recipe,
                seed=record["seed"],
                start=record["task_id"],
                count=1,
                group_size=1,
                max_seconds=data.get("max_seconds", 60),
                max_bytes=data.get("max_bytes", 256 * 1024**2),
                device=device,
                sampling=sampling,
                causal_limits=caller_limits(data) if "causal_limits" in data else None,
            )
            self.task, self.archive, self.run = run_saved(self.output_dir, request)
            self.pair = None
            self.edit = None
            self.intervention = None
            return self._view()
        if self.task is None:
            raise ValueError("generate or open a dataset first")
        if data.get("expected_episode_id", self.task.episode_id) != self.task.episode_id:
            raise ValueError(
                "The active dataset changed in another view. Reopen or replay this dataset first."
            )
        from r0.modules.explorer.lib.observations import observation_action

        observed = observation_action(
            action, self.task, data, self.output_dir, archive=self.archive
        )
        if observed is not None:
            return observed
        from r0.modules.explorer.lib.convolutions import convolution_action

        convolution = convolution_action(action, self.task, data, self.output_dir)
        if convolution is not None:
            return convolution
        if action == "intervention_controls":
            return intervention_controls(self.task)
        if action in ("intervene", "sample_intervention_policy"):
            result = guarded_intervention(
                self.output_dir,
                "sample_policy" if action == "sample_intervention_policy" else "create",
                _identity_numbers(data, for_browser=False),
                before=self.task,
                archive=self.archive,
            )
            self.intervention = saved_intervention(self.output_dir, result)
            return self._view()
        if action in ("intervention_graph", "intervention_export"):
            if self.intervention is None:
                raise ValueError("create or reopen a causal intervention comparison first")
            if data.get("expected_intervention_id") != self.intervention.after.episode_id:
                raise ValueError("The causal comparison changed in another view. Reopen it first.")
            result = guarded_intervention(
                self.output_dir,
                "inspect" if action == "intervention_graph" else "export",
                data,
                pair_path=self.intervention.path,
            )
            if action == "intervention_export":
                return intervention_download(self.output_dir, result)
            return _identity_numbers(
                {"before": result["before"], "after": result["after"]}, for_browser=True
            )
        if action in ("network_noise_draft", "network_draft", "mechanism"):
            from r0.modules.explorer.lib.networks import network_editor_action

            result = network_editor_action(
                action, data, self.task, self.output_dir, limits=caller_limits(data)
            )
            if result is not None:
                return result
        if action == "mechanism":
            return prepare_mechanism_editor(self.task, data, self.output_dir)
        if action == "edit":
            if self.archive is None:
                raise ValueError("save the active dataset before editing")
            edit = create_edit(
                self.output_dir, self.task, self.archive, data, limits=caller_limits(data)
            )
            self.edit = edit
            return self._view()
        if action == "pair":
            if self.archive is None:
                raise ValueError("save the active dataset before comparing")
            pair = create_pair(self.output_dir, self.task, self.archive, self.run, data)
            self.pair = pair
            return self._view()
        if action == "edit_export":
            if self.edit is None:
                raise ValueError("create or open an edit first")
            edit_id = self.edit.after.episode_id
            if data.get("expected_edit_id", edit_id) != edit_id:
                raise ValueError("The active edit changed in another view. Reopen it first.")
            pair = SavedPair(self.edit.before, self.edit.after, self.edit.document, self.edit.path)
            bundle = pair_bundle(pair, max_bytes=data.get("max_bytes", 256 * 1024**2))
            return {
                "name": "mechanism-edit.zip",
                "data": base64.b64encode(bundle).decode("ascii"),
                "mime": "application/zip",
            }
        if action == "pair_export":
            if self.pair is None:
                raise ValueError("create or open a comparison first")
            pair_id = self.pair.document["card"]["pair_id"]
            if data.get("expected_pair_id", pair_id) != pair_id:
                raise ValueError("The active comparison changed in another view. Reopen it first.")
            bundle = pair_bundle(self.pair, max_bytes=data.get("max_bytes", 256 * 1024**2))
            return {
                "name": f"comparison-{pair_id}.zip",
                "data": base64.b64encode(bundle).decode("ascii"),
                "mime": "application/zip",
            }
        if action == "row":
            return {"row": inspect_rows(self.task, [data["index"]])[0]}
        if action == "parameters":
            return {"parameters": json.loads(self.task.mechanism_json)}
        if action == "graph":
            from r0.modules.explorer.lib.network_graph import inspect_graph

            graph = inspect_graph(self.task, self.archive, self.output_dir, data)
            return {"graph": _identity_numbers(graph, for_browser=True)}
        if action == "gp_curve":
            data = _identity_numbers(data, for_browser=False)
            curve = inspect_gp_curve(
                self.task,
                feature=data["feature"],
                anchor_row=data.get("anchor_row", 0),
                points=data.get("points", 129),
                bounds=data.get("bounds"),
                draws=data.get("draws", 0),
                seed=data.get("seed"),
                allow_rank_truncation=data.get("allow_rank_truncation", False),
            )
            return {"curve": _identity_numbers(curve, for_browser=True)}
        if action == "curve":
            return {"curve": inspect_response_curve(self.task, data, self.output_dir)}
        if action == "export":
            bundle = replay_bundle(self.task, max_bytes=data.get("max_bytes", 256 * 1024**2))
            return {
                "name": f"dataset-{self.task.source_task_id}.zip",
                "data": base64.b64encode(bundle).decode("ascii"),
                "mime": "application/zip",
            }
        raise ValueError(f"unknown explorer action: {action}")
