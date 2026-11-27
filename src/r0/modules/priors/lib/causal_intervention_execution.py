"""Surgical runtime hooks: explicit cut locations and retained original-row traces."""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_records import (
    CausalInterventionRecipe,
    HardIntervention,
)
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.representatives import encode, record_fingerprint


class InterventionExecution:
    def __init__(self, recipe: CausalInterventionRecipe, *, baseline: bool = False) -> None:
        self.recipe = recipe
        self.baseline = baseline
        self.traces: dict[int, dict[str, torch.Tensor]] = {}
        self.bypassed: list[int] = []
        self.stream_initial: dict[str, str] = {}

    def rng(
        self, original: Callable[..., np.random.Generator], *keys: object
    ) -> np.random.Generator:
        # Original row positions/split stay aligned even for a fresh population.
        if not self.baseline and self.recipe.mode == "population" and keys[:1] != ("split",):
            result = original("intervention-population-v1", self.recipe.population, *keys)
        else:
            result = original(*keys)
        self.stream_initial[encode(keys)] = record_fingerprint(result.bit_generator.state)
        return result

    def changes(self, index: int) -> tuple[HardIntervention, ...]:
        if self.baseline:
            return ()
        return tuple(
            c
            for c in self.recipe.interventions
            if isinstance(c, HardIntervention) and c.node == index
        )

    def bypass(
        self,
        index: int,
        node: FunctionalCausalNode,
        rows: int,
        device: str,
        *,
        capture_readout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        changes = self.changes(index)
        if not changes or any(c.category is not None for c in changes):
            return None
        width = node.function.output_width
        captured: set[int] = (
            {c.channel for c in node.pipeline.converters}
            if node.pipeline and capture_readout
            else set()
        )
        states: dict[int, float] = {}
        readouts: dict[int, float] = {}
        for c in sorted(changes, key=lambda c: c.location == "readout"):
            for channel, value in zip(c.channels, c.values, strict=True):
                if c.location != "readout":
                    states[channel] = value
                    if channel not in captured:
                        readouts[channel] = value
                if c.location != "state":
                    readouts[channel] = value
        if len(states) != width or len(readouts) != width:
            return None
        self.bypassed.append(index)
        state = (
            torch.tensor([states[c] for c in range(width)], dtype=torch.float64, device=device)
            .expand(rows, width)
            .clone()
        )
        readout = (
            torch.tensor([readouts[c] for c in range(width)], dtype=torch.float64, device=device)
            .expand(rows, width)
            .clone()
        )
        self.record(index, state=state, readout=readout)
        return state, readout

    def apply(
        self, index: int, node: FunctionalCausalNode, state: torch.Tensor, readout: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state, readout = state.clone(), readout.clone()
        captured: set[int] = (
            {c.channel for c in node.pipeline.converters} if node.pipeline else set()
        )
        for change in sorted(self.changes(index), key=lambda c: c.location == "readout"):
            if change.category is not None:
                continue
            for channel, value in zip(change.channels, change.values, strict=True):
                if change.location != "readout":
                    state[:, channel] = value
                    if channel not in captured:
                        readout[:, channel] = value
                if change.location != "state":
                    readout[:, channel] = value
        self.record(index, state=state, readout=readout)
        return state, readout

    def category(self, index: int) -> str | None:
        return next((c.category for c in self.changes(index) if c.category is not None), None)

    def record(self, index: int, **values: torch.Tensor) -> None:
        self.traces.setdefault(index, {}).update(
            {name: value.detach().clone() for name, value in values.items()}
        )

    def metadata(self) -> dict[str, Any]:
        result = {
            "changes": [asdict(c) for c in self.recipe.interventions],
            "mode": self.recipe.mode,
            "population": self.recipe.population if self.recipe.mode == "population" else None,
            "bypassed_nodes": self.bypassed,
            "stream_initial_fingerprints": self.stream_initial,
            "partial_channel_semantics": (
                "coupled equation and pipeline evaluated; selected locations then overwritten"
            ),
            "captured_readouts": (
                "state changes preserve earlier converter captures unless "
                "state_and_readout is explicit"
            ),
            "native_forced_scores": "forced-category indicator; not original equation logits",
            "calibration": (
                "original frozen mechanism and independent target calibration; no query fitting"
            ),
            "unaffected_draws": (
                "original keyed streams in matched mode; bypassed equations consume no draws"
            ),
        }
        from r0.modules.priors.lib.native_vector_records import NativeVectorRecipe

        if isinstance(self.recipe.effective_runtime(), NativeVectorRecipe):
            result["vector_semantics"] = {
                "clamp": "logical emitted value and entire encoded descendant-state span",
                "partial": "one joint equation and correlated innovation evaluated before clamps",
                "full": "all logical variables clamped: joint equation and emission draws bypassed",
                "forced_score": (
                    "threshold: category index in original single score slot; "
                    "other discrete: indicator in original K slots; not original logits"
                ),
                "forced_probability": "separate K-wide category indicator",
            }
        return result
