"""Source-qualified roots compose with every standalone architecture family."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p

ARCHITECTURES = (
    "dense",
    "individual-sparse",
    "block-sparse",
    "skip",
    "residual",
    "irregular",
    "gated",
)
COORDINATES = (
    "gaussian",
    "presampled_gaussian",
    "unit_uniform",
    "symmetric_uniform",
    "categorical",
    "zipf_clipped",
    "zipf_unbounded",
)


def source_root_examples() -> dict[str, Any]:
    profiles: list[tuple[str, p.SourceRootProfile]] = []
    for selector in ("p1-code-normal", "p1-code-uniform", "p1-code-mixed"):
        for presampled in (False, True):
            gaussian = "presampled" if presampled else "standard"
            profiles.append(
                (
                    f"{selector} / {gaussian} Gaussian parameters",
                    p.SourceRootProfile(2, selector=selector, presample_gaussian=presampled),
                )
            )
    profiles.append(
        (
            "P2 node normal/uniform reconstruction",
            p.SourceRootProfile(2, selector="p2-node-mixture-r0"),
        )
    )
    for family in COORDINATES:
        profiles.append(
            (
                f"forced {family} (r0)",
                p.SourceRootProfile(2, selector="forced-r0", families=(family, family)),
            )
        )
    profiles.append(
        (
            "heterogeneous seven-coordinate mixture (r0)",
            p.SourceRootProfile(7, selector="forced-r0", families=COORDINATES),
        )
    )
    examples: dict[str, Any] = {}
    for label, profile in profiles:
        task = p.LinearRecipe(
            rows=64,
            context_rows=48,
            features=profile.width,
            signal_features=profile.width,
            active_features=profile.width,
        )
        for family in ARCHITECTURES:
            request = asdict(p.NetworkSamplingRequest(profile.width, (6, 4, 1), families=(family,)))
            request.pop("limits")
            examples[f"Source root: {label} / {family}"] = {
                "topology": "source-root-network",
                "source": {
                    "topology": "network",
                    "task": asdict(task),
                    "network_prior": request,
                    "root": None,
                },
                "profile": asdict(profile),
                "modes": ["identity"] * profile.width,
                "calibration_rows": 0,
                "root_seed": "7",
                "raw_limits": asdict(p.RawDrawLimits()),
            }
    for mode in ("center", "standardize"):
        source = dict(examples["Source root: heterogeneous seven-coordinate mixture (r0) / gated"])
        source.update(modes=[mode] * 7, calibration_rows=256)
        examples[f"Source root: heterogeneous / gated / independent {mode} (r0)"] = source
    return examples
