"""Explicit source-qualified policies for selecting propagated network channels."""

from dataclasses import dataclass

from r0.modules.priors.lib.functional_graph_admission import admit_functional_graph
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.network_innovation_compile import compile_network_innovations
from r0.modules.priors.lib.network_innovation_plan import NetworkInnovationPlan
from r0.modules.priors.lib.network_innovation_recipe import NetworkInnovationRecipe
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec


@dataclass(frozen=True, slots=True)
class NetworkObservationRequest:
    source: NetworkRecipe | NetworkInnovationRecipe
    features: int
    outputs: int = 1
    profile: str = "p1-paper-block-r0-v1"
    selection: str = "block"
    target: str = "internal"
    feature_order: str = "preserved"
    raw_limits: RawDrawLimits = RawDrawLimits()
    version: str = "network-observation-request-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "network-observation-request-v1"
            or type(self.source) not in (NetworkRecipe, NetworkInnovationRecipe)
            or type(self.features) is not int
            or self.features < 1
            or type(self.outputs) is not int
            or self.outputs < 1
            or self.profile
            not in ("p1-paper-block-r0-v1", "p1-778f0d0-v1", "r0-propagated-block-v1")
            or self.selection not in ("block", "unrestricted")
            or self.target not in ("internal", "final")
            or self.feature_order not in ("preserved", "permuted")
            or type(self.raw_limits) is not RawDrawLimits
        ):
            raise ValueError("invalid network observation policy")
        source, _ = observation_source(self)
        blocks = source.function.blocks
        extension = self.profile == "r0-propagated-block-v1"
        if not extension and any(
            b.parents != (i,) or len(b.experts) != 1 or b.gate or b.skips
            for i, b in enumerate(blocks)
        ):
            raise ValueError("source observation profiles require sequential single-expert layers")
        widths = tuple(b.output_width for b in blocks)
        if not extension and len(set(widths)) != 1:
            raise ValueError("variable-width observation profiles remain a separate extension")
        if self.profile == "p1-778f0d0-v1" and len(blocks) < 2:
            raise ValueError("pinned observation catalog excludes the first affine output")
        count = sum(observation_widths(self))
        if self.features + self.outputs > count - int(
            self.profile == "p1-778f0d0-v1" and self.selection == "unrestricted"
        ):
            raise ValueError("insufficient observation channels; no width repair")
        if self.target == "final" and self.outputs > widths[-1]:
            raise ValueError("final targets must fit the final layer")
        self.admit()

    def admit(self) -> None:
        source, _ = observation_source(self)
        limits = source.limits
        count = sum(observation_widths(self))
        parameters, storage = function_record_cost(self, limits)
        if count > limits.total_channels or parameters + 256 + 12 * count > limits.parameters:
            raise ValueError("network observation catalog exceeds parameter/channel limits")
        _, plan = observation_source(self)
        nodes, _ = compile_network_innovations(source, plan)
        report = admit_functional_graph(
            nodes,
            limits,
            rows=source.task.rows,
            metadata=(self, nodes),
            features=self.features,
            outputs=self.outputs,
        )
        reserved = 4 * storage + 16384 + 2048 * count + 2 * self.raw_limits.state_bytes
        if report["estimated_working_bytes"] + reserved > limits.working_bytes:
            raise ValueError("network observation catalog and snapshots exceed working limit")
        ledger = RawDrawLedger(limits=self.raw_limits)
        ledger.admit_specs(observation_draw_specs(self))


def observation_source(
    request: NetworkObservationRequest,
) -> tuple[NetworkRecipe, NetworkInnovationPlan]:
    if isinstance(request.source, NetworkInnovationRecipe):
        assert request.source.source is not None
        assert request.source.plan is not None
        return request.source.source, request.source.plan
    source = request.source
    return source, NetworkInnovationPlan((None,) * len(source.function.blocks))


def observation_widths(request: NetworkObservationRequest) -> tuple[int, ...]:
    source, _ = observation_source(request)
    widths = tuple(b.output_width for b in source.function.blocks)
    return widths[1:] if request.profile == "p1-778f0d0-v1" else widths


def paper_windows(request: NetworkObservationRequest) -> tuple[tuple[int, int], ...]:
    """(catalog start, number of windows), uniformly weighted over admissible starts."""
    widths = observation_widths(request)
    offset = 0
    result: list[tuple[int, int]] = []
    for i, width in enumerate(widths):
        available = width - (
            request.outputs if request.target == "final" and i == len(widths) - 1 else 0
        )
        windows = max(0, available - request.features + 1)
        result.append((offset, windows))
        offset += width
    if not sum(n for _, n in result):
        raise ValueError("no within-layer feature block fits; no fallback")
    return tuple(result)


def observation_draw_specs(request: NetworkObservationRequest) -> tuple[RawDrawSpec, ...]:
    count = sum(observation_widths(request))
    owner = ("network-observation",)

    def permutation(role: str, n: int) -> RawDrawSpec:
        return RawDrawSpec(owner, role, "permutation", (n,), (("count", n),))

    def index(role: str, n: int) -> RawDrawSpec:
        return RawDrawSpec(owner, role, "integers", (1,), (("high", n),))

    if request.profile == "p1-778f0d0-v1":
        size = request.features + request.outputs
        return (
            (index("window", count - size + 1), permutation("selection", size))
            if request.selection == "block"
            else (permutation("selection", count - 1),)
        )
    if request.selection == "unrestricted":
        return (
            permutation("selection", count - (request.outputs if request.target == "final" else 0)),
        )
    specs = (index("window", sum(n for _, n in paper_windows(request))),)
    if request.target == "internal":
        specs += (permutation("targets", count - request.features),)
    if request.feature_order == "permuted":
        specs += (permutation("feature-order", request.features),)
    return specs
