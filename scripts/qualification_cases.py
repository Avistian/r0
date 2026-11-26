"""Fixed timing workloads; these are not the broad exploration distributions."""

from dataclasses import asdict, replace

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
)

FAMILIES = (
    "linear",
    "grouped",
    "polynomial",
    "quadratic",
    "rff",
    "periodic",
    "radial",
    "sinusoidal",
    "gp",
    "composition",
)
SHAPES = {"small": (64, 8, 4), "medium": (1024, 32, 16), "large": (20000, 256, 256)}
TASK_TYPES = ("regression", "binary", "multiclass")


def fixed_recipe(family, size, task_type):
    rows, features, active = SHAPES[size]
    if family == "gp":
        rows = {"small": 64, "medium": 512, "large": 1024}[size]
    if family == "periodic":
        active = min(active, 64)
    task = LinearRecipe(
        rows=rows,
        context_rows=rows * 3 // 4,
        features=features,
        signal_features=features,
        active_features=active,
        task_type=task_type,
        classes=0
        if task_type == "regression"
        else 2
        if task_type == "binary"
        else 100
        if size == "large"
        else 3,
        noise_ratio=0.5 if task_type == "regression" else 0,
    )
    if family == "linear":
        return task
    if family == "grouped":
        return GroupedLinearRecipe(task=task, group_size=4, calibration_rows=1024)
    if family == "polynomial":
        return PolynomialRecipe(task=task, degree=4, calibration_rows=1024)
    if family == "quadratic":
        return QuadraticRecipe(task=task, rank=32, calibration_rows=1024)
    if family in ("rff", "periodic"):
        return FourierRecipe(
            task=task, family=family, terms=256 if family == "rff" else 64, calibration_rows=1024
        )
    if family in ("radial", "sinusoidal"):
        return SmoothRecipe(task=task, family=family, terms=max(64, active), calibration_rows=1024)
    if family == "gp":
        return GaussianProcessRecipe(task=task, calibration_rows=512)
    if family == "composition":
        # Child observations are discarded; retain full active dimensionality but only 16 rows.
        child = replace(task, rows=16, context_rows=8, noise_ratio=0)
        return CompositeRecipe(
            task=task,
            calibration_rows=1024,
            components=(
                FunctionComponent(child),
                FunctionComponent(PolynomialRecipe(task=child, degree=2, calibration_rows=256)),
                FunctionComponent(
                    SmoothRecipe(task=child, projection="ridge", terms=64, calibration_rows=256)
                ),
            ),
            expression=Combination("sum", (0, Combination("product", (1, 2)))),
        )
    raise ValueError(f"unknown family: {family}")


def cases(families, sizes, task_types, repetitions, seconds, max_bytes):
    result = []
    for family in families:
        for size in sizes:
            for task_type in task_types:
                recipe = fixed_recipe(family, size, task_type)
                for repetition in range(repetitions):
                    result.append(
                        {
                            "id": f"{family}-{size}-{task_type}-{repetition}",
                            "family": family,
                            "size": size,
                            "task_type": task_type,
                            "repetition": repetition,
                            "request": {
                                "recipe": asdict(recipe),
                                "seed": 27,
                                "start": repetition,
                                "count": 1,
                                "group_size": 1,
                                "max_seconds": seconds,
                                "max_bytes": max_bytes,
                                "device": "cpu",
                            },
                        }
                    )
    return result
