"""Compact construction provenance for the explorer's active frozen dataset."""

from typing import Any


def construction_details(recipe: dict[str, Any]) -> dict[str, Any] | None:
    if recipe.get("version") == "network-observation-recipe-v1":
        return {
            "kind": "source-qualified-network-observations",
            "notice": "Full source and selection receipt retained. Selected internal channels "
            "are observations, not original network inputs. Pinned overlap may leak targets; "
            "aliases preserve it explicitly. Frozen Next preserves source/selection draws. "
            "Coordinate-root row-shape RNG behavior is source-dependent. Variable-width and "
            "nonsequential observation policies remain required. Causal references clamp "
            "observed channels while fixing hidden roots and structural innovations.",
            "receipt": recipe["observations"],
        }
    if recipe.get("version") == "standalone-network-innovations-v1":
        from r0.modules.explorer.lib.networks import network_details

        return {
            **network_details(recipe["source"]),
            "kind": "standalone-network-innovations",
            "innovation_plan": recipe["plan"],
            "statistics_sampling": recipe["sampling"],
            "compiled_node_count": len(recipe["nodes"]),
            "notice": (
                "Original network plus explicit internal row innovations; "
                "fixed masks and statistics stay frozen. Source placement/law profiles "
                "do not claim full prior reproduction."
            ),
        }
    if recipe.get("version") in (
        "standalone-network-v1",
        "network-input-view-v1",
        "network-input-view-v2",
    ):
        from r0.modules.explorer.lib.networks import network_details

        return network_details(recipe)
    if recipe.get("version") in (
        "population-fitted-functional-causal-v1",
        "sampled-population-fitted-functional-causal-v1",
    ):
        return {
            "kind": "explicit-population-fitted-trees",
            "parameter_samples": [
                {
                    "node": node,
                    "actual_plan": sample["plan"],
                    "profile": sample["prior"]["profile"],
                    "laws": sample["prior"]["laws"],
                    "parameter_source": sample["parameter_source"],
                    "row_binding": sample["row_binding"],
                    "lifecycle": sample["lifecycle"],
                    "backend_semantics": sample["backend_semantics"],
                    "unused_parameters": sample["unused_parameters"],
                    "raw_draw_count": len(sample["raw"]),
                }
                for node, sample in recipe.get("parameter_samples", [])
            ],
            "population": recipe["population"],
            "lifecycle": recipe["lifecycle"],
            "fit_seed": str(recipe["fit_seed"]),
            "fit_task_id": str(recipe["fit_task_id"]),
            "notice": "Explicit fit event then frozen replay. Support rows exclude query values; "
            "batch reference includes queries and changes function identity on refit. "
            "Mixed populations are independent. Source settings not disclosed by papers "
            "are explicit r0 choices; no full source reproduction or XGBoost claim.",
            "fits": [
                {
                    "node": f["node"],
                    "actual_plan": f["realization"]["plan"],
                    "mixed": f["realization"]["mixed"],
                    "source_contract": f["realization"]["source_contract"],
                    "cardinalities": f["realization"]["cardinalities"],
                    "backend_semantics": f["realization"]["backend_semantics"],
                    "raw_draw_count": len(f["realization"]["raw_draws"]),
                }
                for f in recipe["fits"]
            ],
        }
    if recipe.get("version") in (
        "fitted-functional-causal-v1",
        "sampled-fitted-functional-causal-v1",
    ):
        return {
            "kind": "independent-fitted-trees",
            "notice": (
                "Trees were fitted once on independent synthetic ancestor rows and targets. "
                "Queries never calibrate or refit them. This is an r0 adaptation; XGBoost and "
                "paper-specific refitting lifecycles remain pending. Classifier components "
                "emit category codes into the enclosing task adapter, not native task labels."
            ),
            "fits": [
                {
                    "node": fit["node"],
                    "family": fit["realization"]["plan"]["family"],
                    "task": fit["realization"]["plan"]["task"],
                    "rows": fit["realization"]["plan"]["rows"],
                    "input_law": fit["realization"]["input_law"],
                    "adaptation": fit["realization"]["adaptation"],
                    "backend_version": fit["realization"]["backend_version"],
                    "backend_seeds": fit["realization"]["backend_seeds"],
                    "combiners": [o["combiner"] for o in fit["realization"]["function"]["outputs"]],
                }
                for fit in recipe["fits"]
            ],
            "parameter_samples": [
                {
                    "node": node,
                    "profile": sample["prior"]["profile"],
                    "source": sample["source"],
                    "actual_plan": sample["plan"],
                    "laws": sample["prior"]["laws"],
                    "pinned_formula": (
                        "depth=min(4,2+floor(Exp(scale=2))); count=min(4,1+floor(Exp(scale=2)))"
                        if sample["prior"]["profile"] == "tabicl-pinned-depth-count-v1"
                        else None
                    ),
                    "backend_semantics": sample["backend_semantics"],
                    "lifecycle": sample["lifecycle"],
                    "unused_parameters": sample["unused_parameters"],
                    "raw_draw_count": len(sample["raw"]),
                }
                for node, sample in recipe.get("parameter_samples", [])
            ],
            "backend_randomness": "seeds/version retained; internal backend draws not captured",
        }
    if recipe.get("version") in ("native-vector-causal-v1", "sampled-native-vector-graph-v1"):
        runtime = recipe.get("recipe", recipe)
        return {
            "kind": "native-vector",
            "notice": (
                "One joint score equation per node; ordered logical variables emit continuous "
                "values or one-hot categorical state. Score width, variable count and encoded "
                "width are distinct. This heterogeneous layout is an r0 extension, not a "
                "claim of exact TabAbax reproduction."
            ),
            "domains": [[d["kind"] for d in n["domains"]] for n in runtime["nodes"]],
            "composition": recipe.get("request", {}).get("composition"),
        }
    if recipe.get("version") in ("converted-causal-v1", "sampled-converted-graph-v1"):
        runtime = recipe.get("recipe", recipe)
        return {
            "kind": "typed-category-converters",
            "notice": (
                "Categories are captured before structural innovation and node importance. "
                "Descendants consume the width-preserving numeric representation. Seven TabICL "
                "converter modes use frozen independent calibration (r0 adaptation). Optional "
                "non-Gaussian innovations are r0 extensions; "
                "frequency biases do not guarantee marginals."
            ),
            "converters": [
                [
                    {
                        "name": c["name"],
                        "mode": c["mode"],
                        "span": [c["start"], c["stop"]],
                        "vocabulary": c["vocabulary"],
                    }
                    for c in node["converters"]
                ]
                for node in runtime["nodes"]
            ],
            "composition": recipe.get("request", {}).get("composition"),
            "innovations": [
                p.get("noise", {"sigma": p["plan"]["sigma"]}) for p in recipe.get("pipelines", [])
            ],
        }
    pipelines = recipe.get("pipelines", [])
    if any(p.get("version") == "r0-innovation-pipeline-realization-v2" for p in pipelines):
        return {
            "kind": "r0-innovation-pipelines",
            "notice": (
                "Explicit r0 innovation laws; not the source Gaussian-only profile. "
                "Response scales use the pre-pipeline function output. Innovations enter "
                "after converter capture and before node importance. Independent fit "
                "populations use final ancestor states; query rows never fit laws."
            ),
            "composition": recipe.get("request", {}).get("composition"),
            "innovations": [p["noise"] for p in pipelines],
        }
    composition = recipe.get("composition", recipe.get("request", {}).get("composition"))
    if composition is None:
        return None
    return {
        "kind": "parent-composition",
        "composition": composition,
        "notice": (
            "Parent functions and their reduction are frozen with shared construction choices. "
            "TabICLv2 separates concat from per-parent aggregation. Mitra2 reduces the joint "
            "nonlinear output together with parent projections; weighted/neural parameter "
            "laws are r0 adaptations. The extension catalog is separately named."
        ),
    }
