"""Replayable preferential and hierarchical topology records with strict receipts."""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits, RawDrawRecord
from r0.modules.priors.lib.structural_graph_algorithm import (
    GraphRewrite,
    construct_structural_graph,
)
from r0.modules.priors.lib.structural_graph_prior import (
    StructuralGraphPrior,
    admit_structural_graph,
)

_LIMITS = CausalLimits()
_RAW_LIMITS = RawDrawLimits()
_OWNER = ("structural-topology",)


@dataclass(frozen=True, slots=True)
class StructuralGraph:
    prior: StructuralGraphPrior
    parents: tuple[tuple[int, ...], ...]
    raw: tuple[RawDrawRecord, ...]
    rewrites: tuple[GraphRewrite, ...]
    limits: CausalLimits = _LIMITS
    version: str = "structural-graph-v1"

    def validate(self) -> None:
        admit_structural_graph(self.prior, self.limits)
        if (
            self.version != "structural-graph-v1"
            or type(self.parents) is not tuple
            or len(self.parents) != self.prior.nodes
            or any(
                type(row) is not tuple
                or any(type(p) is not int or not 0 <= p < node for p in row)
                or tuple(sorted(set(row))) != row
                for node, row in enumerate(self.parents)
            )
            or type(self.raw) is not tuple
            or any(type(record) is not RawDrawRecord for record in self.raw)
            or type(self.rewrites) is not tuple
            or any(type(r) is not GraphRewrite for r in self.rewrites)
        ):
            raise ValueError("structural graph requires frozen ordered records and its version")
        index = 0

        def retained(
            role: str, law: str, shape: tuple[int, ...], parameters: tuple[tuple[str, Any], ...]
        ) -> tuple[float | int, ...]:
            nonlocal index
            if index >= len(self.raw):
                raise ValueError("structural graph raw transcript is incomplete")
            record = self.raw[index]
            if (
                record.owner,
                record.role,
                record.stream,
                record.sequence,
                record.distribution,
                record.shape,
                record.parameters,
            ) != (_OWNER, role, "raw", index, law, shape, parameters):
                raise ValueError("structural graph raw binding differs from algorithm")
            if index and (
                record.generator != self.raw[index - 1].generator
                or record.state_before != self.raw[index - 1].state_after
            ):
                raise ValueError("structural graph raw stream chain differs")
            bound = dict(parameters)["high" if law == "integers" else "count"]
            if any(type(v) is not int or not 0 <= v < bound for v in record.values):
                raise ValueError("structural graph raw choice lies outside its support")
            index += 1
            return record.values

        parents, rewrites = construct_structural_graph(self.prior, retained)
        if index != len(self.raw) or parents != self.parents or rewrites != self.rewrites:
            raise ValueError("structural graph differs from its retained construction")

    def replay(self, rng: np.random.Generator) -> "StructuralGraph":
        self.validate()
        with raw_transactions({"raw": rng}, None) as (streams, _):
            for record in self.raw:
                record.replay(streams["raw"])
            return self


def sample_structural_graph(
    prior: StructuralGraphPrior,
    *,
    rng: np.random.Generator,
    limits: CausalLimits = _LIMITS,
    raw_limits: RawDrawLimits = _RAW_LIMITS,
) -> StructuralGraph:
    """Sample once; resource failures and no-op rewrites never cause retries."""
    admit_structural_graph(prior, limits)
    if type(raw_limits) is not RawDrawLimits:
        raise ValueError("structural graph requires frozen raw limits")
    ledger = RawDrawLedger(
        limits=replace(
            raw_limits, working_bytes=min(raw_limits.working_bytes, limits.working_bytes)
        )
    )
    with ledger.transaction(rng) as (staged, capture):

        def draw(
            role: str, law: str, shape: tuple[int, ...], parameters: tuple[tuple[str, Any], ...]
        ) -> tuple[float | int, ...]:
            values = capture.draw(
                staged,
                owner=_OWNER,
                role=role,
                distribution=law,
                shape=shape,
                parameters=parameters,
            )
            return tuple(int(value) for value in values.ravel())

        parents, rewrites = construct_structural_graph(prior, draw)
        graph = StructuralGraph(prior, parents, capture.freeze(), rewrites, limits)
        graph.validate()
        return graph
