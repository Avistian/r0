"""Dispatch retained per-row populations without reducing their lifecycle envelope."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits


def population_recipe_from_dict(fields: dict[str, Any], *, limits: CausalLimits):
    version = fields.get("version")
    if version == "gp-column-target-link-v1":
        from r0.modules.priors.lib.gp_target_links import gp_column_target_link_from_dict

        return gp_column_target_link_from_dict(fields, limits=limits)
    if version == "gp-column-target-recipe-v1":
        from r0.modules.priors.lib.gp_target_environment import gp_column_target_recipe_from_dict

        return gp_column_target_recipe_from_dict(fields, limits=limits)
    if version == "gp-column-recipe-v1":
        from r0.modules.priors.lib.gp_column_decoding import gp_column_recipe_from_dict

        return gp_column_recipe_from_dict(fields, limits=limits)
    if (
        "parent_rows" in fields
        and "observed_rows" in fields
        and fields.get("version")
        not in ("recursive-population-v1", "recursive-population-intervention-v1")
    ):
        raise ValueError("population archive requires its explicit version")
    if version == "recursive-population-intervention-v1":
        from r0.modules.priors.lib.recursive_population_decoding import (
            recursive_population_intervention_from_dict,
        )

        return recursive_population_intervention_from_dict(fields, limits=limits)
    if version == "recursive-population-v1":
        from r0.modules.priors.lib.recursive_population_decoding import (
            recursive_population_from_dict,
        )

        return recursive_population_from_dict(fields, limits=limits)
    return None
