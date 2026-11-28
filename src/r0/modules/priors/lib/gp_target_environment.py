"""Fixed-identity target environments retain GP comparison custody without forging a population."""

from dataclasses import asdict, dataclass
from typing import Any, cast

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.function_metadata import admit_function_metadata
from r0.modules.priors.lib.gp_column_comparison import (
    GPColumnComparison,
    gp_column_comparison_from_dict,
    gp_comparison_metadata,
)
from r0.modules.priors.lib.gp_column_decoding import gp_fields
from r0.modules.priors.lib.gp_column_metadata import gp_sequence
from r0.modules.priors.lib.gp_column_output import GPColumnRecipe
from r0.modules.priors.lib.gp_column_population import GPColumnRequest, gp_stream
from r0.modules.priors.lib.gp_column_target import GPColumnTarget
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask

_DEFAULT_LIMITS = CausalLimits()


def admit_gp_target_environment(
    request: GPColumnRequest,
    target: GPColumnTarget,
    task: LinearRecipe,
    limits: CausalLimits,
    *,
    extra_numbers: int = 0,
) -> None:
    """Reserve comparison replay plus both task sides and validation tensor copies."""
    GPColumnRecipe(request, task)
    for cap in (request.limits, limits):
        report = request.admit(cap)
        if target.input_width + len(target.weights) + target.outputs > cap.total_channels:
            raise ValueError("GP target environment channel limit exceeded")
        count, storage = function_record_cost(target, cap)
        count += (
            8
            * (request.rows + request.calibration_rows)
            * (target.input_width + len(target.weights) + target.outputs)
        )
        storage += 128 * count
        # Two retained tasks and two replay-validation tasks. Includes inputs,
        # scores, targets, probabilities, split indices and adapter temporaries.
        cells = extra_numbers + 4 * request.rows * (
            request.columns + 4 * request.outputs + task.classes + 4
        )
        if (
            4 * (count + report["parameters"]) + cells > cap.parameters
            or 4 * (storage + report["estimated_working_bytes"]) + 128 * cells > cap.working_bytes
        ):
            raise ValueError("GP target environment aggregate comparison/task limit exceeded")


@dataclass(frozen=True, slots=True)
class GPColumnTargetRecipe:
    comparison: GPColumnComparison
    task: LinearRecipe
    side: str
    version: str = "gp-column-target-recipe-v1"

    def __post_init__(self) -> None:
        if (
            type(self.comparison) is not GPColumnComparison
            or type(self.task) is not LinearRecipe
            or type(self.side) is not str
            or self.side not in ("original", "replacement")
            or self.version != "gp-column-target-recipe-v1"
        ):
            raise ValueError("invalid GP target environment")
        admit_gp_target_environment(
            self.request, self.comparison.replacement, self.task, self.request.limits
        )

    @property
    def request(self) -> GPColumnRequest:
        return self.comparison.original.request


def gp_target_environment_metadata(
    data: Any, *, limits: CausalLimits
) -> tuple[dict[str, Any], LinearRecipe]:
    admit_function_metadata(data, limits)
    raw = gp_fields(data, GPColumnTargetRecipe)
    if type(raw["side"]) is not str or raw["side"] not in ("original", "replacement"):
        raise ValueError("invalid GP target environment side")
    task_raw = gp_fields(raw["task"], LinearRecipe)
    gp_sequence(task_raw["class_offsets"])
    task_raw["class_offsets"] = tuple(task_raw["class_offsets"])
    task = LinearRecipe(**task_raw)
    request, target = gp_comparison_metadata(raw["comparison"], limits=limits)
    admit_gp_target_environment(request, target, task, limits)
    return raw, task


def gp_column_target_recipe_from_dict(
    data: Any, *, limits: CausalLimits = _DEFAULT_LIMITS
) -> GPColumnTargetRecipe:
    raw, task = gp_target_environment_metadata(data, limits=limits)
    return GPColumnTargetRecipe(
        gp_column_comparison_from_dict(raw["comparison"], limits=limits), task, raw["side"]
    )


def gp_target_adapter_stream(recipe: GPColumnTargetRecipe, *roles: object):
    """Both sides share adapter draws; matched reproduces the original adapter exactly."""
    p = recipe.comparison.original
    if recipe.comparison.mode == "matched":
        return gp_stream(p.seed, p.task_id, "adapter", *roles)
    return gp_stream(
        p.seed, p.task_id, "gp-target-comparison-adapter", recipe.comparison.fresh_seed, *roles
    )


def generate_gp_target_environment(
    recipe: GPColumnTargetRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    p = recipe.comparison.original
    if any(type(v) is not int for v in (seed, task_id)) or (seed, task_id) != (p.seed, p.task_id):
        raise ValueError("GP target environment requires its exact retained population identity")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("select an available CPU/CUDA device; no fallback")
    admit_gp_target_environment(
        recipe.request, recipe.comparison.replacement, recipe.task, recipe.request.limits
    )
    # Validate complete receipt before using any supplied arrays, including typed
    # dataclass replacements. This is not a new population or a query refit.
    valid = gp_column_target_recipe_from_dict(asdict(recipe), limits=recipe.request.limits)
    c = valid.comparison
    scores = c.original_scores if valid.side == "original" else c.replacement_scores
    return complete_task(
        valid.task,
        asdict(valid),
        {
            "family": "gp_columns",
            "version": valid.version,
            "kind": "gp-target-edit",
            "side": valid.side,
            "comparison": asdict(c),
            "lifecycle": "fixed original locations/kernels; shared matched or fresh innovations",
            "calibration": "original independent marginal inputs; separate target calibration",
            "query_fit": False,
        },
        torch.tensor(c.draw.values, dtype=torch.float64, device=device),
        torch.tensor(scores, dtype=torch.float64, device=device),
        seed=seed,
        task_id=task_id,
        rng=lambda *role: gp_target_adapter_stream(valid, *role),
    )


def gp_target_record_metadata(record: Any, *, limits: CausalLimits) -> None:
    """Bind transport identity before any numerical receipt replay."""
    if type(record) is not dict:
        raise ValueError("GP environment transport must be an object")
    record = cast(dict[str, Any], record)
    if set(record) != {"recipe", "seed", "task_id", "execution"}:
        raise ValueError("GP environment transport schema differs")
    raw = record["recipe"]
    if type(raw) is dict and cast(dict[str, Any], raw).get("version") == "gp-column-target-link-v1":
        from r0.modules.priors.lib.gp_target_links import gp_target_link_metadata

        base, _, _ = gp_target_link_metadata(raw, limits=limits)
    else:
        base, _ = gp_target_environment_metadata(raw, limits=limits)
    original = base["comparison"]["original"]
    if any(type(record[k]) is not int or record[k] != original[k] for k in ("seed", "task_id")):
        raise ValueError("GP environment transport identity differs")
    execution = cast(dict[str, Any], record["execution"])
    keys = {"implementation", "device", "python", "numpy", "torch", "cpu_threads", "cuda", "gpu"}
    if (
        type(execution) is not dict
        or set(execution) != keys
        or execution["implementation"] != raw["version"]
        or execution["device"] not in ("cpu", "cuda")
        or any(type(execution[k]) is not str for k in ("python", "numpy", "torch"))
        or type(execution["cpu_threads"]) is not int
        or execution["cpu_threads"] < 1
        or any(execution[k] is not None and type(execution[k]) is not str for k in ("cuda", "gpu"))
    ):
        raise ValueError("GP environment execution schema differs")
