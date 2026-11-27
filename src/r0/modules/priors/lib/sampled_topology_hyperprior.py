"""Owned log/Gamma/topology receipts with exact raw replay and atomic publication."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.causal_graph import CausalGraphPrior
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.raw_draw_spec import RawDrawSpec
from r0.modules.priors.lib.recursive_forest_plan import recursive_effective_limits
from r0.modules.priors.lib.topology_hyperprior import (
    TopologyHyperprior,
    TopologyHyperpriorRequest,
    transform_topology_hyperprior,
)
from r0.modules.priors.lib.topology_hyperprior_admission import (
    admit_topology_hyperprior,
    hyperprior_draw_specs,
)
from r0.modules.priors.lib.topology_hyperprior_algorithm import (
    hyperprior_components,
    hyperprior_parents,
)

_LIMITS = CausalLimits()
_RAW_LIMITS = RawDrawLimits()


@dataclass(frozen=True, slots=True)
class SampledTopologyHyperprior:
    request: TopologyHyperpriorRequest
    log_nodes: float | None
    gamma_unit: float
    gamma_value: float
    topology: CausalGraphPrior
    components: tuple[tuple[int, int], ...]
    parents: tuple[tuple[int, ...], ...]
    raw: tuple[RawDrawRecord, ...]
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    limits: CausalLimits = _LIMITS
    raw_limits: RawDrawLimits = _RAW_LIMITS
    algorithm: str = "balanced-one-parent-redirection-v1"
    version: str = "sampled-topology-hyperprior-v1"

    def __post_init__(self) -> None:
        self.validate_structure()

    def validate_structure(self, *, limits: CausalLimits | None = None) -> None:
        effective = recursive_effective_limits(self.limits, limits or self.limits)
        budget = admit_topology_hyperprior(self.request, effective, self.raw_limits)
        numbers, storage = function_record_cost(self, effective)
        if numbers > budget["parameters"] or 8 * storage > budget["estimated_working_bytes"]:
            raise ValueError("complete topology hyperprior record exceeds aggregate limits")
        if (
            self.version != "sampled-topology-hyperprior-v1"
            or self.algorithm != "balanced-one-parent-redirection-v1"
            or type(self.topology) is not CausalGraphPrior
            or type(self.raw) is not tuple
            or not self.raw
            or any(type(r) is not RawDrawRecord for r in self.raw)
            or type(self.initial) is not GeneratorSnapshot
            or type(self.final) is not GeneratorSnapshot
        ):
            raise ValueError("sampled topology requires complete versioned records")
        index = 0
        previous = self.initial.digest

        def retained(spec: RawDrawSpec) -> float | int:
            nonlocal index, previous
            if index >= len(self.raw):
                raise ValueError("topology hyperprior transcript is incomplete")
            record = self.raw[index]
            if not exact_frozen_equal(
                (record.owner, record.role, record.distribution, record.shape, record.parameters),
                (spec.owner, spec.role, spec.distribution, spec.shape, spec.parameters),
            ) or (
                record.sequence != index
                or record.stream != "raw"
                or record.state_before != previous
                or record.generator != self.initial.engine
                or len(record.values) != 1
            ):
                raise ValueError("topology hyperprior raw binding differs")
            value = record.values[0]
            if type(value) is not (int if spec.distribution == "integers" else float):
                raise ValueError("topology hyperprior raw primitive type differs")
            index += 1
            previous = record.state_after
            return value

        specs = hyperprior_draw_specs(self.request)
        log = float(retained(specs[0])) if isinstance(self.request, TopologyHyperprior) else None
        gamma = float(retained(specs[-1]))
        topology = transform_topology_hyperprior(self.request, log_nodes=log, gamma_unit=gamma)
        parents = hyperprior_parents(topology, retained)
        if (
            index != len(self.raw)
            or previous != self.final.digest
            or self.final.engine != self.initial.engine
            or not exact_frozen_equal(
                (
                    self.log_nodes,
                    self.gamma_unit,
                    self.gamma_value,
                    self.topology,
                    self.components,
                    self.parents,
                ),
                (
                    log,
                    gamma,
                    gamma / self.request.gamma_rate,
                    topology,
                    hyperprior_components(topology.nodes, topology.components),
                    parents,
                ),
            )
        ):
            raise ValueError("sampled topology differs from retained log/Gamma/forest construction")

    def replay(self, *, limits: CausalLimits | None = None) -> "SampledTopologyHyperprior":
        self.validate_structure(limits=limits)
        rng = self.initial.restore(byte_limit=self.raw_limits.state_bytes)
        result = sample_topology_hyperprior(
            self.request, rng=rng, limits=self.limits, raw_limits=self.raw_limits
        )
        if not exact_frozen_equal(result, self):
            raise ValueError("sampled topology hyperprior differs from exact construction replay")
        return result


def sample_topology_hyperprior(
    request: TopologyHyperpriorRequest,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _LIMITS,
    raw_limits: RawDrawLimits = _RAW_LIMITS,
) -> SampledTopologyHyperprior:
    admit_topology_hyperprior(request, limits, raw_limits)
    ledger = RawDrawLedger(limits=raw_limits)
    with ledger.transaction(rng) as (owned, capture):
        initial = snapshot_generator(owned, byte_limit=raw_limits.state_bytes)

        def draw(spec: RawDrawSpec) -> float | int:
            value = capture.draw(
                owned,
                owner=spec.owner,
                role=spec.role,
                distribution=spec.distribution,
                shape=spec.shape,
                parameters=spec.parameters,
            )[0]
            return int(value) if spec.distribution == "integers" else float(value)

        specs = hyperprior_draw_specs(request)
        log = float(draw(specs[0])) if isinstance(request, TopologyHyperprior) else None
        gamma = float(draw(specs[-1]))
        topology = transform_topology_hyperprior(request, log_nodes=log, gamma_unit=gamma)
        parents = hyperprior_parents(topology, draw)
        return SampledTopologyHyperprior(
            request,
            log,
            gamma,
            gamma / request.gamma_rate,
            topology,
            hyperprior_components(topology.nodes, topology.components),
            parents,
            capture.freeze(),
            initial,
            snapshot_generator(owned, byte_limit=raw_limits.state_bytes),
            limits,
            raw_limits,
        )
