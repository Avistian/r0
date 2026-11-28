"""Reserve all parent members and reducer support before any caller stream is used."""

from dataclasses import fields, replace

from r0.modules.priors.lib.fixed_parent_topology import FixedParentTopology
from r0.modules.priors.lib.joint_width_plans import JointWidthPrior
from r0.modules.priors.lib.parent_composition_plans import ParentCompositionPrior
from r0.modules.priors.lib.structural_topology_request import StructuralTopologyRequest
from r0.modules.priors.lib.topology_hyperprior import FixedSizeTopologyHyperprior
from r0.modules.priors.lib.typed_topology_admission import maximum_typed_width
from r0.modules.priors.lib.typed_topology_request import (
    TypedParentTopologyRequest,
    TypedTopologyRequest,
)


def admit_typed_parent_topology(request: TypedParentTopologyRequest) -> dict[str, int]:
    if type(request.composition) is not ParentCompositionPrior:
        raise ValueError("typed parent topology requires a complete composition prior")
    n = request.topology.maximum_nodes
    fan_in = (
        max(map(len, request.topology.parents), default=0)
        if isinstance(request.topology, FixedParentTopology)
        else n - 1
        if isinstance(request.topology, StructuralTopologyRequest)
        else 1
    )
    if isinstance(request.topology, FixedParentTopology):
        request.topology.admit(request.limits)
    # The proxy covers concatenated parent state and neural reduction inputs.
    # One joint member, up to fan_in separate functions, and a reducer coexist.
    members = max(1, fan_in) + 2
    width = maximum_typed_width(request) * members
    controls = {f.name: getattr(request, f.name) for f in fields(TypedTopologyRequest)}
    controls.update(
        topology=FixedSizeTopologyHyperprior(n),
        width_prior=JointWidthPrior(1, width + 1),
        version="typed-topology-request-r0-v1",
    )
    priors = [request.function_prior]
    if request.composition.source == "mitra2":
        priors.append(replace(request.function_prior, functions="r0-linear-projection-v1"))
    if request.composition.aggregation == "mlp" or (
        request.composition.aggregation == "sample" and "mlp" in request.composition.choices
    ):
        priors.append(replace(request.function_prior, functions="r0-neural-reducer-v1"))
    reports: list[dict[str, int]] = []
    for prior in priors:
        controls["function_prior"] = prior
        reports.append(TypedTopologyRequest(**controls).admit())
    report = {key: max(item[key] for item in reports) for key in reports[0]}
    for key in (
        "parameters",
        "estimated_working_bytes",
        "raw_numbers",
        "raw_draws",
        "raw_working_bytes",
    ):
        report[key] *= members
    if isinstance(request.topology, StructuralTopologyRequest):
        extra = request.topology.admit(request.limits, request.raw_limits)
        for key, value in extra.items():
            report[key] += value
    if (
        report["parameters"] > request.limits.parameters
        or report["estimated_working_bytes"] > request.limits.working_bytes
        or report["raw_numbers"] > request.raw_limits.numbers
        or report["raw_draws"] > request.raw_limits.draws
        or report["raw_working_bytes"] > request.raw_limits.working_bytes
    ):
        raise ValueError("complete typed parent composition support exceeds limits")
    return report
