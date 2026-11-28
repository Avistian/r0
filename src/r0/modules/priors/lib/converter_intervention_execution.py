"""Converter stage surgery with explicit bypass boundaries and unchanged innovations."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.categorical_converter import CategoricalEmission
from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.causal_intervention_records import CausalInterventionRecipe
from r0.modules.priors.lib.converted_causal_records import (
    ConvertedCausalNode,
)
from r0.modules.priors.lib.converter_intervention import ConverterHardIntervention, forced_emission
from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline, PipelineEvaluation
from r0.modules.priors.lib.signal_noise import signal_noise_scale


class ConverterInterventionExecution(InterventionExecution):
    def __init__(self, recipe: CausalInterventionRecipe, *, baseline: bool = False) -> None:
        super().__init__(recipe, baseline=baseline)
        self.nodes: tuple[ConvertedCausalNode, ...] = ()
        self.emissions: dict[tuple[int, str], CategoricalEmission] = {}
        self.stage_bypassed: list[int] = []
        self.bound_rng: Callable[..., np.random.Generator] | None = None

    def clamp(self, index: int, name: str) -> ConverterHardIntervention | None:
        if self.baseline:
            return None
        return next(
            (
                c
                for c in self.recipe.interventions
                if isinstance(c, ConverterHardIntervention)
                and c.node == index
                and c.capture == name
            ),
            None,
        )

    def retain_emission(self, index: int, name: str, value: CategoricalEmission) -> None:
        self.record(
            index,
            **{
                f"capture:{name}": value.categories,
                f"probabilities:{name}": value.probabilities,
                f"scores:{name}": value.scores,
                f"representation:{name}": value.state,
            },
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
        converted = self.nodes[index]
        if not converted.converters:
            return super().bypass(index, node, rows, device, capture_readout=capture_readout)
        clamps = [self.clamp(index, c.name) for c in converted.converters]
        if any(c is None for c in clamps):
            return None
        width = node.function.output_width
        dummy = torch.zeros((rows, width), dtype=torch.float64, device=device)
        # A final-state cut plus all named category cuts removes the whole mechanism.
        full = super().bypass(index, node, rows, device, capture_readout=capture_readout)
        if full is not None:
            for converter, clamp in zip(converted.converters, clamps, strict=True):
                assert clamp is not None
                # Final-state cuts supersede the unobserved representation. Do not execute
                # a replaced post function just to manufacture an unused diagnostic.
                emission = forced_emission(
                    converter,
                    replace(clamp, location="capture"),
                    full[0][:, converter.start : converter.stop],
                )
                self.emissions[index, converter.name] = emission
                self.record(
                    index,
                    **{
                        f"capture:{converter.name}": emission.categories,
                        f"probabilities:{converter.name}": emission.probabilities,
                        f"scores:{converter.name}": emission.scores,
                    },
                )
            return full
        spans = {i for c in converted.converters for i in range(c.start, c.stop)}
        if (
            len(spans) != width
            or (node.pipeline and node.pipeline.converters)
            or any(c is None or c.location != "representation_and_capture" for c in clamps)
            or node.noise_response.strength
            and any(node.noise_scale)
        ):
            return None
        # Every pre-innovation channel is surgically replaced; no signal-dependent scale
        # or numeric capture needs the original equation or preprocessing stages.
        for converter, clamp in zip(converted.converters, clamps, strict=True):
            assert clamp is not None
            emission = forced_emission(converter, clamp, dummy[:, converter.start : converter.stop])
            self.emissions[index, converter.name] = emission
            dummy[:, converter.start : converter.stop] = emission.state
            self.retain_emission(index, converter.name, emission)
        assert self.bound_rng is not None
        raw = torch.tensor(
            sample_exogenous(node.exogenous, rows, width, self.bound_rng, index),
            dtype=torch.float64,
            device=device,
        )
        scale = signal_noise_scale(
            torch.zeros_like(dummy), node.noise_scale or (0.0,) * width, node.noise_response
        )
        pipeline = PipelineEvaluation(node.pipeline or CausalNodePipeline(width), index, None)
        state, readout = pipeline.finish(dummy + scale * raw)
        self.stage_bypassed.append(index)
        self.record(index, exogenous=raw, scale=scale, **{"converter-state": dummy})
        return self.apply(index, node, state, readout)

    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata(),
            "converter_semantics": {
                "capture": "forced category indicator; descendant representation unchanged",
                "representation_and_capture": (
                    "forced prototype/embedding/index before innovation and importance"
                ),
                "state": "final numerical state clamp preserves earlier category captures",
                "scores": "forced category indicator, not original distances or logits",
                "stage_bypassed_nodes": self.stage_bypassed,
                "stage_bypass": (
                    "equation and preprocessing bypassed; innovation and importance retained"
                ),
                "partial": (
                    "coupled equation evaluated when unmodified channels or "
                    "signal-dependent scale require it"
                ),
            },
        }
