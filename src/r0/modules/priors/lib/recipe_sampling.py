"""Versioned exploration settings, separate from production family mixture weights."""

import math
from dataclasses import asdict, replace

import numpy as np

from r0.modules.priors.lib.causal_sampling import sample_causal_profile
from r0.modules.priors.lib.compositions import (
    Combination,
    CompositeRecipe,
    Expression,
    FunctionComponent,
)
from r0.modules.priors.lib.exact_gp import GaussianProcessRecipe
from r0.modules.priors.lib.fourier import FourierRecipe
from r0.modules.priors.lib.generate import Recipe
from r0.modules.priors.lib.grouped_linear import GroupedLinearRecipe
from r0.modules.priors.lib.inputs import InputDistribution
from r0.modules.priors.lib.kernels import (
    GibbsKernel,
    Kernel,
    RationalQuadraticKernel,
    SpectralKernel,
    StationaryKernel,
)
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.polynomial import PolynomialRecipe
from r0.modules.priors.lib.quadratic import QuadraticRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.sampling import BaseRecipe
from r0.modules.priors.lib.smooth import SmoothRecipe

BROAD_FAMILIES = (
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


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(math.log(low), math.log(high))))


def _inputs(rng: np.random.Generator, degree: int) -> InputDistribution:
    distribution = str(rng.choice(("gaussian", "uniform", "student_t", "mixture")))
    df = float(rng.choice((0.5, 1, 2, 3, 5, 10, 30, 100)))
    # Conditioning on a finite interval is explicit; no clipping or label-based rejection.
    bound = 4.0 if distribution == "student_t" and df <= 2 * degree else None
    return InputDistribution(
        distribution=distribution,
        location=float(rng.uniform(-2, 2)),
        scale=_log_uniform(rng, 0.25, 4),
        df=df,
        mixture_gap=float(rng.uniform(0.5, 4)),
        bound=bound,
    )


def _base(
    family: str,
    task: LinearRecipe,
    rng: np.random.Generator,
    inputs: InputDistribution | None = None,
) -> BaseRecipe:
    calibration = int(rng.choice((256, 1024, 4096)))
    degree = (
        int(rng.integers(1, 17)) if family == "polynomial" else 2 if family == "quadratic" else 1
    )
    inputs = inputs if inputs is not None else _inputs(rng, degree)
    if family in ("linear", "rff", "periodic"):
        if inputs != InputDistribution():
            raise ValueError("linear and Fourier recipes require standard Gaussian inputs")
        if family == "linear":
            return task
        if family == "periodic" and task.active_features > 64:
            raise ValueError("periodic sampling supports at most 64 active features")
        terms = (
            int(rng.choice((64, 128, 256, 512, 1024, 2048, 4096)))
            if family == "rff"
            else int(rng.integers(task.active_features, 65))
        )
        return FourierRecipe(
            task=task,
            family=family,
            terms=terms,
            length_scale=_log_uniform(rng, 0.05, 20),
            calibration_rows=calibration,
        )
    if family == "grouped":
        divisors = [
            n
            for n in range(1, task.active_features + 1)
            if task.active_features % n == 0 and task.signal_features % n == 0
        ]
        return GroupedLinearRecipe(
            task=task,
            inputs=inputs,
            group_size=int(rng.choice(divisors)),
            calibration_rows=calibration,
        )
    if family == "polynomial":
        return PolynomialRecipe(
            task=task, degree=degree, inputs=inputs, calibration_rows=calibration
        )
    if family == "quadratic":
        return QuadraticRecipe(
            task=task,
            inputs=inputs,
            rank=int(rng.integers(1, 257)),
            interaction_strength=float(rng.uniform(-2, 2)),
            linear_strength=_log_uniform(rng, 0.01, 4),
            calibration_rows=calibration,
        )
    if family in ("radial", "sinusoidal"):
        projection = str(rng.choice(("additive", "ridge")))
        minimum = task.active_features if projection == "additive" else 1
        choices = tuple(n for n in (1, 4, 16, 64, 256, 1024, 4096) if n >= minimum)
        return SmoothRecipe(
            task=task,
            inputs=inputs,
            family=family,
            projection=projection,
            terms=int(rng.choice(choices)),
            length_scale=_log_uniform(rng, 0.05, 20),
            width_spread=_log_uniform(rng, 1, 16),
            calibration_rows=calibration,
        )
    raise ValueError("unsupported analytic sampling family")


def _gp(task: LinearRecipe, rng: np.random.Generator) -> GaussianProcessRecipe:
    count = int(rng.integers(1, 65))
    combine = str(rng.choice(("sum", "product")))
    kernels: list[Kernel] = []
    for i in range(count):
        # The first component covers all active inputs; later ones may act on subsets.
        axes = (
            tuple(range(task.active_features))
            if i == 0
            else tuple(
                sorted(
                    int(v)
                    for v in rng.choice(
                        task.active_features,
                        int(rng.integers(1, task.active_features + 1)),
                        replace=False,
                    )
                )
            )
        )
        kind = str(
            rng.choice(
                (
                    "rbf",
                    "matern12",
                    "matern32",
                    "matern52",
                    "rational_quadratic",
                    "spectral",
                    "gibbs",
                )
            )
        )
        scales = tuple(_log_uniform(rng, 0.25, 4) for _ in axes)
        length, weight = (
            _log_uniform(rng, 0.05, 20),
            _log_uniform(rng, 0.1, 10) if combine == "sum" else 1.0,
        )
        if kind == "rational_quadratic":
            kernel: Kernel = RationalQuadraticKernel(
                length_scale=length,
                axes=axes,
                feature_scales=scales,
                weight=weight,
                alpha=_log_uniform(rng, 0.1, 10),
            )
        elif kind == "spectral":
            kernel = SpectralKernel(
                length_scale=length,
                axes=axes,
                feature_scales=scales,
                weight=weight,
                frequencies=tuple(float(rng.uniform(0, 4)) for _ in axes),
            )
        elif kind == "gibbs":
            kernel = GibbsKernel(
                length_scale=length,
                axes=axes,
                feature_scales=scales,
                weight=weight,
                centers=tuple(float(rng.uniform(-2, 2)) for _ in axes),
                transition_scales=tuple(_log_uniform(rng, 0.25, 4) for _ in axes),
                length_slopes=tuple(float(rng.uniform(-math.log(4), math.log(4))) for _ in axes),
                amplitude_slopes=tuple(
                    float(rng.uniform(-math.log(2), math.log(2))) / len(axes) for _ in axes
                ),
            )
        else:
            kernel = StationaryKernel(
                kind=kind, length_scale=length, axes=axes, feature_scales=scales, weight=weight
            )
        kernels.append(kernel)
    return GaussianProcessRecipe(
        task=task,
        inputs=_inputs(rng, 0),
        kernels=tuple(kernels),
        combine=combine,
        calibration_rows=int(rng.choice((32, 128, 256, 512))),
    )


def _tree(indices: tuple[int, ...], depth: int, rng: np.random.Generator) -> Expression:
    if len(indices) == 1:
        return indices[0]
    if depth == 1:
        children: tuple[Expression, ...] = indices
    else:
        cut = int(rng.integers(1, len(indices)))
        children = (_tree(indices[:cut], depth - 1, rng), _tree(indices[cut:], depth - 1, rng))
    return Combination(
        str(rng.choice(("sum", "product"))),
        children,
        strength=float(rng.choice((-1, 1))) * _log_uniform(rng, 0.25, 4),
    )


def _composition(task: LinearRecipe, rng: np.random.Generator) -> CompositeRecipe:
    count = int(rng.integers(1, 65))
    depth = int(rng.integers(1, 9))
    # 16*64 is the maximum growth degree. Unbounded t inputs must have enough moments.
    inputs = InputDistribution() if rng.random() < 0.25 else _inputs(rng, 16 * 64)
    families = ["grouped", "polynomial", "quadratic", "radial", "sinusoidal"]
    if inputs == InputDistribution():
        families.extend(("linear", "rff"))
        if task.active_features <= 64:
            families.append("periodic")
    components: list[FunctionComponent] = []
    for i in range(count):
        active = task.active_features if i == 0 else int(rng.integers(1, task.active_features + 1))
        axes = tuple(
            sorted(int(v) for v in rng.choice(task.active_features, active, replace=False))
        )
        child = replace(task, active_features=active, noise_ratio=0, separation=1, class_offsets=())
        recipe = _base(str(rng.choice(families)), child, rng, inputs)
        strength = float(rng.choice((-1, 1))) * _log_uniform(rng, 0.25, 4)
        components.append(FunctionComponent(recipe, axes=axes, strength=strength))
    return CompositeRecipe(
        task=task,
        inputs=inputs,
        components=tuple(components),
        expression=_tree(tuple(range(count)), depth, rng),
        calibration_rows=int(rng.choice((256, 1024, 4096))),
    )


def _reference(family: str, task: LinearRecipe, rng: np.random.Generator) -> Recipe:
    if family == "rff":
        return FourierRecipe(task=task, terms=256, length_scale=_log_uniform(rng, 0.05, 20))
    if family in ("quadratic", "product") and task.active_features > 20:
        raise ValueError("adapted quadratic/product reference supports at most 20 active inputs")
    if family == "quadratic":
        return QuadraticRecipe(task=task, rank=20)
    child = replace(task, noise_ratio=0, separation=1, class_offsets=())
    if family == "power":
        exponent = int(rng.integers(2, 6))
        return CompositeRecipe(
            task=task,
            components=(FunctionComponent(child),),
            expression=Combination("product", (0,) * exponent),
        )
    components: list[FunctionComponent] = []
    for _ in range(2):
        name = str(rng.choice(("linear", "quadratic", "rff", "radial", "sinusoidal")))
        if name in ("quadratic", "rff"):
            recipe = _reference(name, child, rng)
            assert isinstance(recipe, (QuadraticRecipe, FourierRecipe))
        elif name == "linear":
            recipe = child
        else:
            recipe = SmoothRecipe(
                task=child, family=name, terms=64, length_scale=_log_uniform(rng, 0.05, 20)
            )
        components.append(FunctionComponent(recipe))
    return CompositeRecipe(
        task=task, components=tuple(components), expression=Combination("product", (0, 1))
    )


def sample_recipe(
    family: str,
    *,
    task: LinearRecipe,
    seed: int,
    task_id: int,
    profile: str = "broad-v1",
) -> Recipe:
    """Choose mechanism settings; preserve caller-selected shape and target controls.

    Save the returned ordinary recipe before generation, including failed attempts.
    Exact parameter ranges and compatibility exclusions are documented in
    docs/sampling-profiles.md. This function performs no generation, retry or filtering.
    """
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    families = {
        "broad-v1": BROAD_FAMILIES,
        "tabiclv2-adapted-v1": ("power", "quadratic", "rff", "product"),
        "causal-affine-v1": ("forward", "layered", "redirection"),
        "causal-activated-v1": ("forward", "layered", "redirection"),
        "causal-preferences-v1": ("forward", "layered", "redirection"),
    }
    if profile not in families or family not in families[profile]:
        raise ValueError("unsupported sampling profile or family")
    if profile in ("causal-affine-v1", "causal-activated-v1", "causal-preferences-v1"):
        return sample_causal_profile(
            family,
            task=task,
            seed=seed,
            task_id=task_id,
            profile=profile,
        )
    rng = np.random.default_rng(
        int(record_fingerprint((profile, family, asdict(task), seed, task_id)), 16)
    )
    if profile == "tabiclv2-adapted-v1":
        return _reference(family, task, rng)
    if family == "composition":
        return _composition(task, rng)
    if family == "gp":
        return _gp(task, rng)
    inputs = InputDistribution() if family in ("linear", "rff", "periodic") else None
    return _base(family, task, rng, inputs)
