"""Finite GP feature populations with explicit reconstructed target and kernel laws."""

from dataclasses import asdict
from typing import Any, cast

from r0.modules import priors as p
from r0.modules.episodes import TaskType


def gp_column_examples() -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for family in ("linear", "quadratic", "neural"):
        for task_type, classes in (("regression", 0), ("binary", 2), ("multiclass", 3)):
            for kernel_mode in ("shared RBF", "per-column RBF"):
                task = p.LinearRecipe(
                    rows=16,
                    context_rows=8,
                    features=2,
                    signal_features=2,
                    active_features=2,
                    task_type=cast(TaskType, task_type),
                    classes=classes,
                )
                request = p.GPColumnRequest(
                    rows=16,
                    columns=2,
                    target_family=family,
                    outputs=3 if task_type == "multiclass" else 1,
                    kernels=(p.GPColumnKernel(0.1, 1.0),)
                    if kernel_mode == "shared RBF"
                    else (
                        p.GPColumnKernel(0.1, 1.0),
                        p.GPColumnKernel(0.2, 0.5),
                    ),
                )
                examples[f"GP columns: {family} / {task_type} / {kernel_mode}"] = {
                    "topology": "gp-columns",
                    "task": asdict(task),
                    "request": asdict(request),
                }
    return examples
