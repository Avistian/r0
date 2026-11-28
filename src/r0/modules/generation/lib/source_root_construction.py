"""Guarded root-profile selection and explicitly independent normalization."""

from dataclasses import asdict, replace
from typing import Any, cast

import numpy as np

from r0.modules import priors


def construct_source_root_network(
    fields: dict[str, Any], *, seed: int, limits: priors.CausalLimits
) -> priors.NetworkRecipe:
    from r0.modules.generation.lib.construction import construct_graph

    if set(fields) != {"source", "profile", "modes", "calibration_rows", "root_seed", "raw_limits"}:
        raise ValueError(
            "source-root network requires explicit source/profile/normalization/seed/raw controls"
        )
    source_controls = cast(dict[str, Any], fields["source"])
    if (
        type(source_controls) is not dict
        or source_controls.get("topology")
        not in ("network", "network-authored", "network-hyperprior")
        or source_controls.get("root") is not None
    ):
        raise ValueError("source-root wrapper requires a standalone architecture with root=null")
    profile = priors.source_root_profile_from_dict(fields["profile"])
    if source_controls.get("task", {}).get("features") != profile.width:
        raise ValueError("source-root profile width must match observed network input geometry")
    modes = fields["modes"]
    if (
        type(modes) not in (list, tuple)
        or len(modes) != profile.width
        or any(mode not in ("identity", "center", "standardize") for mode in modes)
    ):
        raise ValueError(
            "source root requires one explicit independent normalization mode per coordinate"
        )
    independent = any(mode != "identity" for mode in modes)
    rows = fields["calibration_rows"]
    if (
        type(rows) is not int
        or (not independent and rows != 0)
        or (independent and rows < (2 if "standardize" in modes else 1))
    ):
        raise ValueError(
            "identity requires zero calibration rows; independent modes require explicit rows"
        )
    root_seed = fields["root_seed"]
    if type(root_seed) is str and root_seed.isascii() and root_seed.isdecimal():
        root_seed = int(root_seed)
    if type(root_seed) is not int or not 0 <= root_seed < 2**63:
        raise ValueError("root seed must be an exact integer in [0,2**63)")
    raw_limits = priors.RawDrawLimits(**fields["raw_limits"])
    source = construct_graph(source_controls, seed=seed, limits=limits)
    if type(source) is not priors.NetworkRecipe:
        raise ValueError("source-root wrapper requires a standalone network recipe")
    model, selection, calibration = (
        np.random.default_rng(s) for s in np.random.SeedSequence(root_seed).spawn(3)
    )
    selected = priors.sample_source_root_profile(
        profile, model_rng=model, selection_rng=selection, limits=limits
    )
    root = priors.sample_source_root_base(
        selected,
        modes=tuple(modes),
        calibration_rows=rows,
        calibration_rng=calibration if independent else None,
        limits=limits,
        raw_limits=raw_limits,
    )
    recipe = replace(source, root=root)
    # Decode the complete archive once inside the guard, checking source/root
    # custody and aggregate resource binding before publishing the recipe.
    verified = priors.recipe_from_dict(asdict(recipe), limits=limits)
    if not isinstance(verified, priors.NetworkRecipe):
        raise ValueError("source-root construction produced an incompatible recipe")
    return verified
