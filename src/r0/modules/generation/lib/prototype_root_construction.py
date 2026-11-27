"""Guarded independent prototype banks retain source and weight-law provenance."""

from dataclasses import asdict, replace
from typing import Any, cast

import numpy as np

from r0.modules import priors as p


def construct_prototype_network(
    fields: dict[str, Any], *, seed: int, limits: p.CausalLimits
) -> p.NetworkRecipe:
    from r0.modules.generation.lib.construction import construct_graph

    if set(fields) != {
        "source",
        "lifecycle",
        "reference_rows",
        "rho",
        "count_rule",
        "max_prototypes",
        "weights",
        "raw_limits",
        "bank_seed",
    }:
        raise ValueError("prototype construction requires explicit bank/lifecycle/weight controls")
    if fields["lifecycle"] != "frozen-independent-prototype-bank-r0-v1":
        raise ValueError(
            "batch-reference prototype workbench is pending; never converted to a frozen bank"
        )
    source_controls = cast(dict[str, Any], fields["source"])
    if type(source_controls) is not dict or source_controls.get("topology") not in (
        "network",
        "network-authored",
        "network-hyperprior",
        "source-root-network",
    ):
        raise ValueError(
            "prototype bank requires a standalone architecture/source-root construction"
        )
    bank_seed = fields["bank_seed"]
    if type(bank_seed) is str and bank_seed.isascii() and bank_seed.isdecimal():
        bank_seed = int(bank_seed)
    if type(bank_seed) is not int or not 0 <= bank_seed < 2**63:
        raise ValueError("prototype bank seed must be an exact integer in [0,2**63)")
    weights = p.prototype_weight_law_from_dict(fields["weights"])
    source = construct_graph(source_controls, seed=seed, limits=limits)
    if type(source) is not p.NetworkRecipe or isinstance(source.root, p.PrototypeRootBase):
        raise ValueError("prototype construction requires a nonprototype root population")
    plan = p.PrototypeRootPlan(
        source.root or p.JointRootBase("gaussian", source.function.input_width),
        fields["reference_rows"],
        fields["rho"],
        weights,
        fields["count_rule"],
        fields["max_prototypes"],
        p.RawDrawLimits(**fields["raw_limits"]),
    )
    sampled = p.sample_prototype_root(plan, rng=np.random.default_rng(bank_seed), limits=limits)
    recipe = replace(source, root=p.PrototypeRootBase(sampled))
    verified = p.recipe_from_dict(asdict(recipe), limits=limits)
    if not isinstance(verified, p.NetworkRecipe):
        raise ValueError("prototype construction returned an incompatible recipe")
    return verified
