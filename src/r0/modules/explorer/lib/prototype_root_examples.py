"""Explicit r0 prototype conventions across all seven architecture templates."""

from dataclasses import asdict
from typing import Any

from r0.modules import priors as p
from r0.modules.explorer.lib.source_root_examples import ARCHITECTURES, source_root_examples


def prototype_root_examples() -> dict[str, Any]:
    sources = source_root_examples()
    examples: dict[str, Any] = {}
    for family in ARCHITECTURES:
        for mode in (
            "multinomial-trial-temperature-r0",
            "multinomial-probability-temperature-r0",
            "dirichlet-concentration-r0",
        ):
            for count_rule in ("ceil-r0", "floor-r0", "nearest-even-r0"):
                examples[f"Prototype root: {family} / {mode} / {count_rule}"] = {
                    "topology": "prototype-root-network",
                    "source": sources[f"Source root: forced unit_uniform (r0) / {family}"],
                    "lifecycle": "frozen-independent-prototype-bank-r0-v1",
                    "reference_rows": 32,
                    "rho": 0.125,
                    "count_rule": count_rule,
                    "max_prototypes": 256,
                    "weights": asdict(
                        p.PrototypeWeightLaw(
                            mode=mode, trials=None if mode == "dirichlet-concentration-r0" else 16
                        )
                    ),
                    "raw_limits": asdict(p.RawDrawLimits()),
                    "bank_seed": "13",
                }
    return examples
