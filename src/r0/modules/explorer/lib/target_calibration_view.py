"""Describe retained independent target fits without treating them as query statistics."""

from typing import Any


def target_calibration_view(recipe: dict[str, Any]) -> dict[str, Any] | None:
    original_version = recipe.get("version")
    while recipe.get("version") in (
        "calibrated-population-query-v1",
        "gp-root-calibration-query-v1",
        "calibrated-population-intervention-v1",
        "gp-root-intervention-v1",
    ):
        recipe = recipe[
            "source"
            if recipe["version"]
            in ("calibrated-population-query-v1", "gp-root-calibration-query-v1")
            else "base"
        ]
    kinds = {
        "gp-root-calibration-v1": "independent GP population",
        "calibrated-population-graph-v1": "independent row-owned population",
    }
    version = recipe.get("version")
    kind = kinds.get(version) if isinstance(version, str) else None
    if kind is None:
        return None
    return {
        "statistics": recipe["calibration"],
        "population_kind": kind,
        "source_version": original_version,
    }
