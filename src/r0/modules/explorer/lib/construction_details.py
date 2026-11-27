"""Compact construction provenance for the explorer's active frozen dataset."""

from typing import Any


def construction_details(recipe: dict[str, Any]) -> dict[str, Any] | None:
    if recipe.get("version") == "fitted-functional-causal-v1":
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
