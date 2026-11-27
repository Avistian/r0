"""Typed captures inserted before innovation through the shared structural evaluator."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from r0.modules.priors.lib.categorical_converter import CategoricalEmission
from r0.modules.priors.lib.converted_causal_records import (
    ConvertedCausalNode,
    ConvertedCausalRecipe,
)
from r0.modules.priors.lib.converter_intervention import forced_emission
from r0.modules.priors.lib.converter_intervention_execution import ConverterInterventionExecution
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_node_evaluation import evaluate_functional_nodes
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline, PipelineEvaluation
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw
from r0.modules.priors.lib.representatives import record_fingerprint


@dataclass(frozen=True, slots=True)
class ConvertedEvaluation:
    states: tuple[torch.Tensor, ...]
    readouts: tuple[torch.Tensor, ...]
    emissions: dict[tuple[int, str], CategoricalEmission]
    uniforms: dict[tuple[int, str], torch.Tensor]


def converted_rng(seed: int, task_id: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(
        int(record_fingerprint(("converted-causal-v1", seed, task_id, keys)), 16)
    )


class ConvertedPipelineEvaluation(PipelineEvaluation):
    def __init__(
        self,
        node: ConvertedCausalNode,
        index: int,
        rng: Callable[..., np.random.Generator],
        emissions: dict[tuple[int, str], CategoricalEmission],
        uniforms: dict[tuple[int, str], torch.Tensor],
        capture: RawDrawLedger | None,
        owner: tuple[str, ...],
        *,
        record_steps: bool,
        execution: ConverterInterventionExecution | None = None,
    ) -> None:
        super().__init__(
            node.equation.pipeline or CausalNodePipeline(node.equation.function.output_width),
            index,
            None,
            record_steps=record_steps,
        )
        self.execution = execution
        self.converted = node
        self.rng = rng
        self.emissions = emissions
        self.uniforms = uniforms
        self.capture = capture
        self.owner = owner

    def before_noise(self, value: torch.Tensor) -> torch.Tensor:
        state = super().before_noise(value)
        numeric_steps = self.steps[-2:] if self.record_steps else []
        if self.record_steps:
            self.steps = self.steps[:-2]
        preprocessing = [f"pipeline-{self.node}-{self.steps[-1][0]}"] if self.steps else []
        representations: list[str] = []
        for converter in self.converted.converters:
            uniform = None
            clamp = self.execution.clamp(self.node, converter.name) if self.execution else None
            needs_original = clamp is None or (
                clamp.location == "capture" and not converter.mode.endswith("_id")
            )
            if converter.stochastic and needs_original:
                uniform = torch.tensor(
                    recorded_draw(
                        self.rng("converter-emission", self.node, converter.name),
                        self.capture,
                        owner=(*self.owner, f"node:{self.node}", "converter", converter.name),
                        role="emission-uniform",
                        distribution="random",
                        shape=(len(state),),
                    ),
                    dtype=state.dtype,
                    device=state.device,
                )
                self.uniforms[self.node, converter.name] = uniform
            values = state[:, converter.start : converter.stop]
            emission = converter.apply(values, uniforms=uniform) if needs_original else None
            if clamp is not None:
                emission = forced_emission(converter, clamp, values, emission)
            assert emission is not None
            if self.execution:
                self.execution.retain_emission(self.node, converter.name, emission)
                if uniform is not None:
                    self.execution.record(self.node, **{f"uniform:{converter.name}": uniform})
            self.emissions[self.node, converter.name] = emission
            state[:, converter.start : converter.stop] = emission.state
            if self.record_steps:
                prefix = f"category-{converter.name}"

                def retain_stage(
                    suffix: str, tensor: torch.Tensor, parents: list[str], prefix: str = prefix
                ) -> str:
                    name = f"{prefix}-{suffix}"
                    self.retain(name, tensor)
                    self.trace_dependencies[name] = parents
                    return f"pipeline-{self.node}-{name}"

                score = retain_stage("scores", emission.scores, preprocessing)
                probability = retain_stage("probabilities", emission.probabilities, [score])
                emission_parents = [probability]
                if uniform is not None:
                    emission_parents.append(retain_stage("uniforms", uniform[:, None], []))
                category = retain_stage("capture", emission.categories[:, None], emission_parents)
                representations.append(
                    retain_stage(
                        "representation",
                        emission.state,
                        preprocessing if converter.mode.endswith("_id") else [category],
                    )
                )
        if self.record_steps:
            self.steps += [
                (name, state.clone() if name == "converter-state" else tensor)
                for name, tensor in numeric_steps
            ]
            self.trace_dependencies["converter-state"] = [*preprocessing, *representations]
            self.trace_dependencies["converted-readout"] = preprocessing
        return state


def evaluate_converted_nodes(
    nodes: tuple[ConvertedCausalNode, ...],
    rows: int,
    rng: Callable[..., np.random.Generator],
    device: str,
    *,
    capture: RawDrawLedger | None = None,
    owner: tuple[str, ...] = (),
    add: Callable[..., None] | None = None,
    indices: Sequence[int] = (),
    execution: ConverterInterventionExecution | None = None,
) -> ConvertedEvaluation:
    emissions: dict[tuple[int, str], CategoricalEmission] = {}
    uniforms: dict[tuple[int, str], torch.Tensor] = {}
    readouts: list[torch.Tensor] = []
    if execution:
        execution.nodes, execution.emissions, execution.bound_rng = nodes, emissions, rng

    def factory(index: int, equation: FunctionalCausalNode) -> PipelineEvaluation:
        return ConvertedPipelineEvaluation(
            nodes[index],
            index,
            rng,
            emissions,
            uniforms,
            capture,
            owner,
            record_steps=add is not None,
            execution=execution,
        )

    states = evaluate_functional_nodes(
        tuple(n.equation for n in nodes),
        rows,
        rng,
        device,
        readouts=readouts,
        capture=capture,
        owner=owner,
        add=add,
        indices=indices,
        pipeline_factory=factory,
        execution=execution,
    )
    return ConvertedEvaluation(tuple(states), tuple(readouts), emissions, uniforms)


def evaluate_converted(
    recipe: ConvertedCausalRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
    add: Callable[..., None] | None = None,
    indices: Sequence[int] = (),
    execution: ConverterInterventionExecution | None = None,
) -> ConvertedEvaluation:
    if type(recipe) is not ConvertedCausalRecipe or any(
        type(i) is not int or not 0 <= i < 2**63 for i in (seed, task_id)
    ):
        raise ValueError("converted graph requires frozen recipe and bounded seed/task identity")
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("converted graph device unavailable")
    recipe.admit()

    def original_rng(*keys: object) -> np.random.Generator:
        return converted_rng(seed, task_id, *keys)

    def rng(*keys: object) -> np.random.Generator:
        return execution.rng(original_rng, *keys) if execution else original_rng(*keys)

    return evaluate_converted_nodes(
        recipe.nodes,
        recipe.task.rows,
        rng,
        device,
        add=add,
        indices=indices,
        execution=execution,
    )
