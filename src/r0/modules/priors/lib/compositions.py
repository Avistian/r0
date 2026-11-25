"""Explicit sums and products of analytic functions evaluated on shared inputs."""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.calibration import stable_calibration
from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe
from r0.modules.priors.lib.fourier import FourierRecipe
from r0.modules.priors.lib.inputs import InputDistribution, sample_inputs
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.polynomial import PolynomialRecipe
from r0.modules.priors.lib.quadratic import QuadraticRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.sampling import BaseRecipe, generate_base_task
from r0.modules.priors.lib.signals import composite_scores
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


def _strength(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value == 0:
        raise ValueError("component and combination strengths must be finite and nonzero")


@dataclass(frozen=True, slots=True)
class FunctionComponent:
    recipe: BaseRecipe
    axes: tuple[int, ...] = ()
    strength: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.recipe, GaussianProcessRecipe):
            raise ValueError(
                "function compositions require analytic components; direct GP is a reference"
            )
        if (
            type(self.axes) is not tuple
            or any(type(v) is not int or v < 0 for v in self.axes)
            or len(set(self.axes)) != len(self.axes)
        ):
            raise ValueError("component axes must be unique nonnegative integers")
        _strength(self.strength)


@dataclass(frozen=True, slots=True)
class Combination:
    operation: str
    children: tuple["Expression", ...]
    strength: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.operation not in ("sum", "product")
            or type(self.children) is not tuple
            or not 2 <= len(self.children) <= 64
        ):
            raise ValueError("combinations need sum/product and 2..64 children")
        _strength(self.strength)


type Expression = int | Combination


def _controls(recipe: BaseRecipe) -> LinearRecipe:
    return recipe if isinstance(recipe, LinearRecipe) else recipe.task


def _degree(recipe: BaseRecipe) -> int:
    if isinstance(recipe, PolynomialRecipe):
        return recipe.degree
    if isinstance(recipe, QuadraticRecipe):
        return 2
    return 1 if isinstance(recipe, LinearRecipe) or recipe.version == "grouped-linear-v1" else 0


def _expression_properties(
    expression: Expression, components: tuple[FunctionComponent, ...], depth: int = 0
) -> tuple[list[int], int, int]:
    if type(expression) is int:
        if not 0 <= expression < len(components):
            raise ValueError("expression references a missing component")
        return [expression], 1, _degree(components[expression].recipe)
    if not isinstance(expression, Combination) or depth >= 8:
        raise ValueError("expression depth must be at most 8")
    children = [
        _expression_properties(child, components, depth + 1) for child in expression.children
    ]
    used = [i for indices, _, _ in children for i in indices]
    if len(used) > 64:
        raise ValueError("expression may contain at most 64 component occurrences")
    combine = sum if expression.operation == "product" else max
    orders = [order for _, order, _ in children]
    degrees = [degree for _, _, degree in children]
    return used, combine(orders), combine(degrees)


@dataclass(frozen=True, slots=True)
class CompositeRecipe:
    components: tuple[FunctionComponent, ...]
    task: LinearRecipe = field(default_factory=LinearRecipe)
    inputs: InputDistribution = field(default_factory=InputDistribution)
    expression: Expression = 0
    calibration_rows: int = 4096
    version: str = "composite-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "composite-v1"
            or type(self.components) is not tuple
            or not 1 <= len(self.components) <= 64
        ):
            raise ValueError("composite-v1 needs 1..64 components")
        if type(self.calibration_rows) is not int or not 256 <= self.calibration_rows <= 16384:
            raise ValueError("calibration rows must be 256..16384")
        used, _, degree = _expression_properties(self.expression, self.components)
        if set(used) != set(range(len(self.components))):
            raise ValueError("expression must use every declared component")
        axes_used: set[int] = set()
        for component in self.components:
            axes = component.axes or tuple(range(self.task.active_features))
            controls = _controls(component.recipe)
            if len(axes) != controls.active_features or any(
                v >= self.task.active_features for v in axes
            ):
                raise ValueError(
                    "component axes must match its active count within parent active features"
                )
            if (controls.task_type, controls.classes) != (self.task.task_type, self.task.classes):
                raise ValueError("component output types and class counts must match the parent")
            if (
                controls.noise_ratio != 0
                or controls.separation != 1
                or any(offset != 0 for offset in controls.class_offsets)
            ):
                raise ValueError(
                    "observation noise and classification controls belong on the parent"
                )
            distribution = (
                InputDistribution()
                if isinstance(component.recipe, (LinearRecipe, FourierRecipe))
                else component.recipe.inputs
            )
            if distribution != self.inputs:
                raise ValueError("components must use the parent's input distribution")
            axes_used.update(axes)
        if axes_used != set(range(self.task.active_features)):
            raise ValueError("components must cover every parent active feature")
        if (
            self.inputs.distribution == "student_t"
            and self.inputs.bound is None
            and self.inputs.df <= 2 * degree
        ):
            raise ValueError(
                "composition variance requires input moment 2*growth_degree: df must exceed it"
            )


def generate_composite(
    recipe: CompositeRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("select an available CPU/CUDA device; no fallback")

    def rng(*keys: object) -> np.random.Generator:
        return np.random.default_rng(
            int(record_fingerprint((recipe.version, seed, task_id, keys)), 16)
        )

    task = recipe.task
    active = rng("active").permutation(task.signal_features)[: task.active_features]
    components: list[dict[str, Any]] = []
    for i, component in enumerate(recipe.components):
        child_seed = int(rng("component", i).integers(0, 2**63))
        child = generate_base_task(component.recipe, seed=child_seed, task_id=0, device="cpu")
        axes = component.axes or tuple(range(task.active_features))
        components.append(
            {
                "mechanism": json.loads(child.mechanism_json),
                "feature_count": child.inputs.shape[1],
                "input_columns": active[list(axes)].tolist(),
                "strength": component.strength,
                "seed": child_seed,
            }
        )
    _, order, degree = _expression_properties(recipe.expression, recipe.components)
    expression = (
        asdict(recipe.expression)
        if isinstance(recipe.expression, Combination)
        else recipe.expression
    )
    mechanism: dict[str, Any] = {
        "family": "composite",
        "version": recipe.version,
        "active": active.tolist(),
        "components": components,
        "expression": expression,
        "interaction_order": order,
        "growth_degree": degree,
        "inputs": asdict(recipe.inputs),
        "calibration_rows": recipe.calibration_rows,
        "normalization": "estimated from independent shared calibration inputs",
    }
    calibration = sample_inputs(
        recipe.inputs, recipe.calibration_rows, task.features, rng, "calibration"
    )
    raw = composite_scores(mechanism, torch.tensor(calibration))
    center, scale = stable_calibration(np.asarray(raw.numpy(), dtype=np.float64))
    mechanism.update({"center": center.tolist(), "scale": scale.tolist()})
    inputs = torch.tensor(
        sample_inputs(recipe.inputs, task.rows, task.features, rng, "input"), device=device
    )
    scores = (
        composite_scores(mechanism, inputs) - torch.tensor(center, device=device)
    ) / torch.tensor(scale, device=device)
    config = asdict(recipe)
    source_config = asdict(recipe)
    for component_config in source_config["components"]:
        child_config = component_config["recipe"]
        child_controls = (
            child_config if child_config["version"] == "linear-v1" else child_config["task"]
        )
        child_controls.pop("split", None)
        child_controls.pop("context_rows", None)
    return complete_task(
        task,
        config,
        mechanism,
        inputs,
        scores,
        seed=seed,
        task_id=task_id,
        rng=rng,
        source_config=source_config,
    )
