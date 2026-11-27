"""Worst endpoint graph/raw/snapshot reservation before any topology RNG access."""

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.topology_hyperprior import (
    FixedSizeTopologyHyperprior,
    TopologyHyperprior,
    TopologyHyperpriorRequest,
)

OWNER = ("topology-hyperprior",)


def hyperprior_draw_specs(request: TopologyHyperpriorRequest) -> tuple[RawDrawSpec, ...]:
    specs: list[RawDrawSpec] = []
    if isinstance(request, TopologyHyperprior):
        specs.append(
            RawDrawSpec(
                OWNER,
                "log-size",
                "uniform",
                (1,),
                (("low", request.log_nodes_low), ("high", request.log_nodes_high)),
            )
        )
    specs.append(
        RawDrawSpec(OWNER, "gamma-unit", "gamma", (1,), (("concentration", request.gamma_shape),))
    )
    return tuple(specs)


def topology_edge_specs(node: int, start: int) -> tuple[RawDrawSpec, RawDrawSpec]:
    owner = (*OWNER, f"node:{node}")
    return (
        RawDrawSpec(owner, "predecessor", "integers", (1,), (("high", node - start),)),
        RawDrawSpec(owner, "redirect", "random", (1,)),
    )


def admit_topology_hyperprior(
    request: TopologyHyperpriorRequest, limits: CausalLimits, raw_limits: RawDrawLimits
) -> dict[str, int]:
    if (
        type(request) not in (TopologyHyperprior, FixedSizeTopologyHyperprior)
        or type(limits) is not CausalLimits
        or type(raw_limits) is not RawDrawLimits
    ):
        raise ValueError("topology hyperprior admission requires explicit frozen controls")
    n = request.maximum_nodes
    if n > min(limits.nodes, limits.total_channels):
        raise ValueError("topology hyperprior worst size exceeds node/channel limits")
    numbers, storage = function_record_cost(request, limits)
    parameters = numbers + 64 * n + 2048
    working = 8 * storage + 65536 * n + 32 * raw_limits.state_bytes + 2 * 1024**2
    if parameters > limits.parameters or working > limits.working_bytes:
        raise ValueError("topology hyperprior worst geometry/metadata exceeds limits")
    scalar_specs = hyperprior_draw_specs(request)
    edge_specs = topology_edge_specs(n - 1, 0) if n > request.components else ()
    count = len(scalar_specs) + 2 * (n - request.components)
    retained = sum(spec.retained_bytes("raw") for spec in scalar_specs)
    retained += (n - request.components) * sum(spec.retained_bytes("raw") for spec in edge_specs)
    if (
        count > raw_limits.draws
        or count > raw_limits.numbers
        or retained + 2 * raw_limits.state_bytes > raw_limits.working_bytes
    ):
        raise ValueError("topology hyperprior worst raw record limits exceeded")
    # Bound every header by the largest node index, before allocating O(N) declarations.
    specs = (*scalar_specs, *(edge_specs * (n - request.components)))
    ledger = RawDrawLedger(limits=raw_limits)
    ledger.admit_specs(tuple(specs))
    return {
        "maximum_nodes": n,
        "maximum_edges": n - request.components,
        "parameters": parameters,
        "estimated_working_bytes": working,
        "draws": len(specs),
        "numbers": sum(s.numbers for s in specs),
    }
