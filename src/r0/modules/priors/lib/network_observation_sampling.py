"""Retained observation selections bind actual compiled propagated block states."""

from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.exact_frozen_record import exact_frozen_equal
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.generator_snapshot import GeneratorSnapshot, snapshot_generator
from r0.modules.priors.lib.joint_function_admission import function_record_cost
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_innovation_compile import compile_network_innovations
from r0.modules.priors.lib.network_observation_request import (
    NetworkObservationRequest,
    observation_draw_specs,
    observation_source,
    observation_widths,
    paper_windows,
)
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord


@dataclass(frozen=True, slots=True)
class NetworkObservationChannel:
    block: int
    node: int
    channel: int
    stage: str = "propagated-block-state"

    def __post_init__(self) -> None:
        if (
            any(type(v) is not int or v < 0 for v in (self.block, self.node, self.channel))
            or self.stage != "propagated-block-state"
        ):
            raise ValueError("invalid network observation channel")


def network_observation_catalog(
    request: NetworkObservationRequest,
) -> tuple[NetworkObservationChannel, ...]:
    request.admit()
    source, plan = observation_source(request)
    _, mapping = compile_network_innovations(source, plan)
    first = int(request.profile == "p1-778f0d0-v1")
    return tuple(
        NetworkObservationChannel(i, mapping[i + 1], j)
        for i in range(first, len(source.function.blocks))
        for j in range(source.function.blocks[i].output_width)
    )


def observation_indices(
    request: NetworkObservationRequest, raw: tuple[RawDrawRecord, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    draws = {r.role: tuple(int(v) for v in r.values) for r in raw}
    count = sum(observation_widths(request))
    pinned = request.profile == "p1-778f0d0-v1"
    final = tuple(range(count - request.outputs, count))
    if pinned or request.selection == "unrestricted":
        start = draws["window"][0] if pinned and request.selection == "block" else 0
        order = tuple(start + i for i in draws["selection"])
        targets = final if request.target == "final" else order[: request.outputs]
        # Pinned code reserves target slots even when final targets override them.
        skip = request.outputs if pinned or request.target == "internal" else 0
        features = order[skip : skip + request.features]
    else:
        selected = draws["window"][0]
        start = 0
        for offset, windows in paper_windows(request):
            if selected < windows:
                start = offset + selected
                break
            selected -= windows
        features = tuple(range(start, start + request.features))
        available = tuple(i for i in range(count) if i not in features)
        targets = (
            final
            if request.target == "final"
            else tuple(available[i] for i in draws["targets"][: request.outputs])
        )
        if request.feature_order == "permuted":
            features = tuple(features[i] for i in draws["feature-order"])
    if request.feature_order == "preserved":
        features = tuple(sorted(features))
    return features, targets


def _draw(
    request: NetworkObservationRequest, rng: np.random.Generator, ledger: RawDrawLedger
) -> None:
    for spec in observation_draw_specs(request):
        ledger.draw(
            rng,
            owner=spec.owner,
            role=spec.role,
            distribution=spec.distribution,
            shape=spec.shape,
            parameters=spec.parameters,
        )


@dataclass(frozen=True, slots=True)
class SampledNetworkObservations:
    request: NetworkObservationRequest
    catalog: tuple[NetworkObservationChannel, ...]
    features: tuple[int, ...]
    targets: tuple[int, ...]
    initial: GeneratorSnapshot
    final: GeneratorSnapshot
    raw: tuple[RawDrawRecord, ...]
    version: str = "sampled-network-observations-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "sampled-network-observations-v1"
            or type(self.request) is not NetworkObservationRequest
        ):
            raise ValueError("invalid sampled network observation envelope")
        self.validate_replay()

    @property
    def target_feature_overlap(self) -> tuple[int, ...]:
        return tuple(i for i in self.features if i in self.targets)

    def validate_replay(self) -> None:
        self.request.admit()
        source, _ = observation_source(self.request)
        function_record_cost(self, source.limits)
        if not exact_frozen_equal(self.catalog, network_observation_catalog(self.request)):
            raise ValueError("observation catalog differs from compiled stage references")
        if type(self.initial) is not GeneratorSnapshot or type(self.final) is not GeneratorSnapshot:
            raise ValueError("observation replay requires frozen generator snapshots")
        ledger = RawDrawLedger(limits=self.request.raw_limits)
        ledger.admit_specs(observation_draw_specs(self.request))
        rng = self.initial.restore(byte_limit=self.request.raw_limits.state_bytes)
        with ledger.transaction(rng) as (active, capture):
            _draw(self.request, active, capture)
            raw = capture.freeze()
            indices = observation_indices(self.request, raw)
            final = snapshot_generator(active, byte_limit=self.request.raw_limits.state_bytes)
            if not exact_frozen_equal(
                (raw, indices, final), (self.raw, (self.features, self.targets), self.final)
            ):
                raise ValueError("network observation raw draws, indices or final state differ")

    def runtime(self, task: LinearRecipe) -> FunctionalCausalRecipe:
        """Explicit task adapter; keep this envelope separately until recipe integration."""
        from r0.modules.priors.lib.network_observation_runtime import (
            materialize_network_observations,
        )

        return materialize_network_observations(self, task)


def sample_network_observations(
    request: NetworkObservationRequest, *, rng: np.random.Generator
) -> SampledNetworkObservations:
    if type(request) is not NetworkObservationRequest:
        raise ValueError("network observation sampling requires its explicit request")
    request.admit()
    catalog = network_observation_catalog(request)
    ledger = RawDrawLedger(limits=request.raw_limits)
    ledger.admit_specs(observation_draw_specs(request))
    with ledger.transaction(rng) as (active, capture):
        initial = snapshot_generator(active, byte_limit=request.raw_limits.state_bytes)
        _draw(request, active, capture)
        raw = capture.freeze()
        features, targets = observation_indices(request, raw)
        return SampledNetworkObservations(
            request,
            catalog,
            features,
            targets,
            initial,
            snapshot_generator(active, byte_limit=request.raw_limits.state_bytes),
            raw,
        )
