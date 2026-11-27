"""Raw source and normalization records remain visible without parent-side replay."""

from typing import Any


def source_root_details(root: dict[str, Any] | None) -> dict[str, Any] | None:
    if root is None or root.get("version") not in (
        "source-root-base-v1",
        "sampled-source-root-base-v1",
    ):
        return None
    sampled = root["version"] == "sampled-source-root-base-v1"
    profile = root.get("profile")
    return {
        "kind": "source-coordinate-root",
        "version": root["version"],
        "law": root["law"],
        "calibration": root["calibration"],
        "sampling": profile,
        "independent_calibration": {
            "modes": root.get("modes"),
            "rows": root.get("calibration_rows", 0),
            "raw": root.get("calibration_raw", []),
            "initial": root.get("calibration_initial"),
            "final": root.get("calibration_final"),
        }
        if sampled
        else None,
        "notice": (
            "Frozen source-coordinate laws. Sampled parameters and independent normalization "
            "remain in the archive. Independent rows do not use displayed query rows and do not "
            "reproduce batch normalization. P1 paper multivariate inputs are underspecified; "
            "the pinned categorical branch is a distinct option. P2 node mixture is a named r0 "
            "reconstruction, not shared-prototype coupling."
        ),
    }
