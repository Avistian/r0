"""Distinct explicit population recipe versions share strict archive dispatch."""

from typing import Any

from r0.modules.priors.lib.causal_limits import CausalLimits


def population_recipe_from_dict(fields: dict[str, Any], *, limits: CausalLimits):
    version = fields.get("version", "linear-v1")
    if version == "population-xgboost-graph-v1":
        from r0.modules.priors.lib.xgboost_population_graph_decoding import (
            xgboost_population_graph_from_dict,
        )

        return xgboost_population_graph_from_dict(fields, limits=limits)
    if version == "gp-root-calibration-v1":
        from r0.modules.priors.lib.gp_root_calibration_decoding import gp_root_calibration_from_dict

        return gp_root_calibration_from_dict(fields, limits=limits)
    if version == "gp-root-dag-v1":
        from r0.modules.priors.lib.gp_root_decoding import gp_root_dag_from_dict

        return gp_root_dag_from_dict(fields, limits=limits)
    if version == "network-input-view-v2":
        from r0.modules.priors.lib.matched_network_view import matched_view_from_dict

        return matched_view_from_dict(fields, limits=limits)
    if version == "network-input-view-v1":
        from r0.modules.priors.lib.network_input_view import network_view_from_dict

        return network_view_from_dict(fields, limits=limits)
    if version == "sampled-joint-hyperprior-graph-v1":
        from r0.modules.priors.lib.joint_hyperprior_decoding import joint_hyperprior_graph_from_dict

        return joint_hyperprior_graph_from_dict(fields, limits=limits)
    if version == "sampled-hyperprior-graph-v1":
        from r0.modules.priors.lib.hyperprior_graph_decoding import hyperprior_graph_from_dict

        return hyperprior_graph_from_dict(fields, limits=limits)
    if version == "calibrated-population-query-v1":
        from r0.modules.priors.lib.calibrated_query_decoding import calibrated_query_from_dict

        return calibrated_query_from_dict(fields, limits=limits)
    if version == "calibrated-population-intervention-v1":
        from r0.modules.priors.lib.calibrated_population_intervention_decoding import (
            calibrated_intervention_from_dict,
        )

        return calibrated_intervention_from_dict(fields, limits=limits)
    if version == "calibrated-population-graph-v1":
        from r0.modules.priors.lib.population_calibration_decoding import (
            calibrated_population_from_dict,
        )

        return calibrated_population_from_dict(fields, limits=limits)
    if version == "network-observation-recipe-v1":
        from r0.modules.priors.lib.network_observation_recipe_decoding import (
            network_observation_recipe_from_dict,
        )

        return network_observation_recipe_from_dict(fields, limits=limits)
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
    if (
        "bindings" in fields
        and "request" in fields
        and fields.get("version") != "recursive-population-graph-v1"
    ):
        raise ValueError("population DAG archive requires its explicit version")
    if {"base", "change", "samples", "inputs"} <= fields.keys() and fields.get(
        "version"
    ) != "recursive-population-graph-intervention-v1":
        raise ValueError("population DAG intervention requires its explicit version")
    version = fields.get("version", "linear-v1")
    if version == "gp-root-dag-v1":
        from r0.modules.priors.lib.gp_root_decoding import gp_root_dag_from_dict

        return gp_root_dag_from_dict(fields, limits=limits)
    if version == "network-input-view-v2":
        from r0.modules.priors.lib.matched_network_view import matched_view_from_dict

        return matched_view_from_dict(fields, limits=limits)
    if version == "network-input-view-v1":
        from r0.modules.priors.lib.network_input_view import network_view_from_dict

        return network_view_from_dict(fields, limits=limits)
    if version == "sampled-joint-hyperprior-graph-v1":
        from r0.modules.priors.lib.joint_hyperprior_decoding import joint_hyperprior_graph_from_dict

        return joint_hyperprior_graph_from_dict(fields, limits=limits)
    if version == "sampled-hyperprior-graph-v1":
        from r0.modules.priors.lib.hyperprior_graph_decoding import hyperprior_graph_from_dict

        return hyperprior_graph_from_dict(fields, limits=limits)
    if version == "network-observation-recipe-v1":
        from r0.modules.priors.lib.network_observation_recipe_decoding import (
            network_observation_recipe_from_dict,
        )

        return network_observation_recipe_from_dict(fields, limits=limits)
    if version == "gp-column-target-link-v1":
        from r0.modules.priors.lib.gp_target_links import gp_column_target_link_from_dict

        return gp_column_target_link_from_dict(fields, limits=limits)
    if version == "recursive-population-graph-intervention-v1":
        from r0.modules.priors.lib.population_graph_intervention_decoding import (
            population_graph_intervention_from_dict,
        )

        return population_graph_intervention_from_dict(fields, limits=limits)
    if version == "recursive-population-graph-v1":
        from r0.modules.priors.lib.recursive_population_graph_decoding import (
            recursive_population_graph_from_dict,
        )

        return recursive_population_graph_from_dict(fields, limits=limits)
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
