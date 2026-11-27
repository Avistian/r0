"""Two-stream frozen law selection, retaining conditional and deliberately unused draws."""

from typing import Any, cast

import numpy as np

from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.source_root_laws import SourceRootCoordinate, SourceRootLaw
from r0.modules.priors.lib.source_root_profiles import SourceRootProfile


def realize_root_profile(
    profile: SourceRootProfile,
    model: np.random.Generator,
    selection: np.random.Generator,
    ledger: RawDrawLedger,
) -> tuple[SourceRootLaw, tuple[SourceRootCoordinate, ...], tuple[float, ...]]:
    # Selection has the longer stream name, so this bounds both streams together.
    ledger.admit_draws(selection, profile.worst_draws())

    def draw(
        rng: np.random.Generator,
        owner: tuple[str, ...],
        role: str,
        distribution: str = "random",
        shape: tuple[int, ...] = (),
        parameters: tuple[tuple[str, Any], ...] = (),
    ) -> Any:
        return ledger.draw(
            rng,
            owner=owner,
            role=role,
            distribution=distribution,
            shape=shape,
            parameters=parameters,
        )

    gaussians: tuple[SourceRootCoordinate, ...] = ()
    if profile.gaussian_width:
        means = draw(
            model, ("root-profile", "model"), "gaussian_means", "normal", (profile.gaussian_width,)
        )
        factors = draw(
            model,
            ("root-profile", "model"),
            "gaussian_scale_factors",
            "normal",
            (profile.gaussian_width,),
        )
        gaussians = tuple(
            SourceRootCoordinate("gaussian", location=float(mu), scale=abs(float(mu) * float(z)))
            for mu, z in zip(means, factors, strict=True)
        )
    thresholds: tuple[float, ...] = ()
    normal_threshold, categorical_threshold = 0.0, 0.0
    owner = ("root-profile", "selection")
    if profile.selector == "p1-code-mixed":
        thresholds = tuple(
            0.66 * float(draw(selection, owner, role))
            for role in ("zipf_threshold_unused", "categorical_threshold", "normal_threshold")
        )
    if thresholds:
        categorical_threshold, normal_threshold = thresholds[1], thresholds[2]
    node_family = ""
    if profile.selector == "p2-node-mixture-r0":
        u = float(draw(selection, owner, "node_family"))
        node_family = "gaussian" if u < profile.p2_normal_probability else "symmetric_uniform"
    laws: list[SourceRootCoordinate] = []
    gaussian_index = 0
    for i in range(profile.width):
        owner = ("root-profile", f"coordinate:{i}")
        if profile.selector == "forced-r0":
            family = profile.families[i]
        elif profile.selector == "p2-node-mixture-r0":
            family = node_family
        elif profile.selector == "p1-code-uniform":
            family = "unit_uniform"
        elif (
            profile.selector == "p1-code-normal"
            or float(draw(selection, owner, "normal_test")) > normal_threshold
        ):
            family = "gaussian"
        elif float(draw(selection, owner, "categorical_test")) > categorical_threshold:
            family = "categorical"
        else:
            family = "zipf_clipped"
        if family == "gaussian" and profile.presample_gaussian:
            laws.append(gaussians[i])
        elif family == "presampled_gaussian":
            laws.append(gaussians[gaussian_index])
            gaussian_index += 1
        elif family == "categorical":
            count = 2 + int(
                draw(selection, owner, "category_count", "integers", parameters=(("high", 9),))
            )
            weights = draw(selection, owner, "category_weights", shape=(count,))
            total = float(cast(Any, np.sum(weights)))
            if not np.isfinite(total) or total <= 0:
                raise ValueError("degenerate category weights; no retry")
            laws.append(
                SourceRootCoordinate(
                    "categorical", probabilities=tuple(float(v) for v in weights / total)
                )
            )
        elif family == "zipf_clipped":
            exponent = 2 + 2 * float(draw(selection, owner, "zipf_exponent"))
            laws.append(SourceRootCoordinate("zipf", exponent=exponent, clip=10))
        elif family == "zipf_unbounded":
            laws.append(SourceRootCoordinate("zipf", exponent=profile.unbounded_exponent))
        else:
            laws.append(SourceRootCoordinate(family))
    return SourceRootLaw(tuple(laws)), gaussians, thresholds
