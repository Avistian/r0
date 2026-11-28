"""Source-specific multi-parent compositions with explicit editable graph custody."""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from r0.modules import priors as p
from r0.modules.explorer.lib.typed_topology_examples import typed_topology_examples


def typed_parent_topology_examples() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in ("tabiclv2", "mitra2", "extension"):
        for example in typed_topology_examples().values():
            controls = deepcopy(example)
            request = controls["request"]
            request.update(
                version="typed-parent-topology-request-r0-v1",
                topology=asdict(p.FixedParentTopology(((), (0,), (0, 1), (1, 2)))),
                composition=asdict(
                    p.ParentCompositionPrior(
                        source=source,
                        mode="sample" if source == "mitra2" else "aggregate",
                        aggregation="max",
                    )
                ),
                limits=asdict(p.CausalLimits(parameters=2_000_000_000, working_bytes=1024**4)),
                raw_limits=asdict(
                    p.RawDrawLimits(numbers=2_000_000_000, draws=2_000_000, working_bytes=1024**4)
                ),
            )
            result[f"Parent topology: {source} / {request['family']}"] = controls
    return result
