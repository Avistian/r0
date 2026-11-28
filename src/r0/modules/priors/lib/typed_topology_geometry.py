"""Native semantic layout and independent log-width draws with retained raw custody."""

import math
from dataclasses import dataclass

import numpy as np

from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.raw_draw_records import RawDrawRecord
from r0.modules.priors.lib.typed_topology_request import TypedTopologyRequest


@dataclass(frozen=True, slots=True)
class NativeTopologyGeometry:
    widths: tuple[int, ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[int, int]
    raw: tuple[RawDrawRecord, ...]
    version: str = "native-topology-geometry-r0-v1"

    def validate(self, request: TypedTopologyRequest, nodes: int) -> None:
        if self.version != "native-topology-geometry-r0-v1" or request.native is None:
            raise ValueError("native geometry requires its request and version")
        shuffled = request.native.placement == "permuted-distinct-nodes-r0-v1"
        if type(self.raw) is not tuple or len(self.raw) != 1 + shuffled:
            raise ValueError("native geometry requires complete width/placement raw custody")
        for i, record in enumerate(self.raw):
            permutation = i == 1
            if (
                type(record) is not RawDrawRecord
                or record.owner != ("typed-topology", "native-geometry")
                or record.role != ("placement" if permutation else "width-quantiles")
                or record.distribution != ("permutation" if permutation else "random")
                or record.shape != (nodes,)
                or record.parameters != ((("count", nodes),) if permutation else ())
                or record.stream != "raw"
                or record.sequence != i
                or record.dtype != ("<i8" if permutation else "<f8")
                or i
                and (
                    record.state_before != self.raw[i - 1].state_after
                    or record.generator != self.raw[i - 1].generator
                )
            ):
                raise ValueError("native geometry raw law or stream binding differs")
        values = self.raw[0].values
        if any(type(q) is not float or not 0 <= q < 1 for q in values):
            raise ValueError("native width quantiles require finite unit-interval floats")
        expected = tuple(
            math.floor(
                math.exp(
                    math.log(request.width_prior.low)
                    + q * math.log(request.width_prior.high / request.width_prior.low)
                )
            )
            for q in values
        )
        order = self.raw[1].values if shuffled else (*range(nodes - 1), nodes - 1)
        if tuple(sorted(order)) != tuple(range(nodes)) or any(type(i) is not int for i in order):
            raise ValueError("native readout placement is not an exact node permutation")
        observed = tuple((i, 0) for i in order[: request.task.features])
        target = (order[-1], 0)
        if (
            type(self.widths) is not tuple
            or any(type(w) is not int for w in self.widths)
            or self.widths != expected
            or any(not request.width_prior.low <= w < request.width_prior.high for w in self.widths)
            or type(self.observed) is not tuple
            or any(
                type(ref) is not tuple or len(ref) != 2 or any(type(v) is not int for v in ref)
                for ref in (*self.observed, self.target)
            )
            or type(self.target) is not tuple
            or any(type(v) is not int for v in self.target)
            or self.observed != observed
            or self.target != target
        ):
            raise ValueError("native geometry differs from its raw width/readout laws")


def sample_native_topology_geometry(
    request: TypedTopologyRequest, nodes: int, rng: np.random.Generator
) -> NativeTopologyGeometry:
    assert request.native is not None
    ledger = RawDrawLedger(limits=request.raw_limits)
    with ledger.transaction(rng) as (stream, capture):
        quantiles = capture.draw(
            stream,
            owner=("typed-topology", "native-geometry"),
            role="width-quantiles",
            distribution="random",
            shape=(nodes,),
        )
        widths = tuple(
            math.floor(
                math.exp(
                    math.log(request.width_prior.low)
                    + float(q) * math.log(request.width_prior.high / request.width_prior.low)
                )
            )
            for q in quantiles
        )
        order = tuple(range(nodes))
        if request.native.placement == "permuted-distinct-nodes-r0-v1":
            order = tuple(
                int(v)
                for v in capture.draw(
                    stream,
                    owner=("typed-topology", "native-geometry"),
                    role="placement",
                    distribution="permutation",
                    shape=(nodes,),
                    parameters=(("count", nodes),),
                )
            )
        result = NativeTopologyGeometry(
            widths,
            tuple((i, 0) for i in order[: request.task.features]),
            (order[-1], 0),
            capture.freeze(),
        )
        result.validate(request, nodes)
        return result
