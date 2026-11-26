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
    sample_recipe,
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
REFERENCE_FAMILIES = ("power", "quadratic", "rff", "product")


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


def sampled_cases(profile, families, sizes, task_types, repetitions, seconds, max_bytes):
    """Predeclare independent profile draws; no success-based selection or retries."""
    result = []
    for family in families:
        for size in sizes:
            rows, features, active = {
                "small": (16, 1, 1),
                "medium": (512, 32, 8),
                "large": (20000, 256, 256),
            }[size]
            if family == "gp":
                rows = {"small": 16, "medium": 512, "large": 1024}[size]
            if family == "periodic":
                active = min(active, 64)
            if profile == "tabiclv2-adapted-v1" and family in ("quadratic", "product"):
                active = min(active, 20)
            for task_type in task_types:
                classes = (
                    0
                    if task_type == "regression"
                    else 2
                    if task_type == "binary"
                    else (100 if size == "large" else 3)
                )
                offsets = ()
                if size == "large" and task_type != "regression":
                    offsets = (
                        (2.0,)
                        if classes == 2
                        else tuple(-2 + 4 * i / (classes - 1) for i in range(classes))
                    )
                controls = LinearRecipe(
                    rows=rows,
                    context_rows=1 if size == "small" else rows * 3 // 4,
                    features=features,
                    signal_features=features,
                    active_features=active,
                    task_type=task_type,
                    classes=classes,
                    noise_ratio={"small": 0, "medium": 0.5, "large": 2}[size]
                    if task_type == "regression"
                    else 0,
                    separation={"small": 0.1, "medium": 1, "large": 4}[size]
                    if task_type != "regression"
                    else 1,
                    class_offsets=offsets,
                )
                for repetition in range(repetitions):
                    recipe = sample_recipe(
                        family, task=controls, seed=127, task_id=repetition, profile=profile
                    )
                    result.append(
                        {
                            "id": f"{family}-{size}-{task_type}-{repetition}",
                            "family": family,
                            "size": size,
                            "task_type": task_type,
                            "repetition": repetition,
                            "selection": {
                                "profile": profile,
                                "family": family,
                                "task": asdict(controls),
                                "seed": 127,
                                "task_id": repetition,
                            },
                            "request": {
                                "recipe": asdict(recipe),
                                "seed": 127,
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
