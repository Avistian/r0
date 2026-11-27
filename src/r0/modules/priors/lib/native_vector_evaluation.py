"""Evaluate one joint score equation per node, then ordered heterogeneous emissions."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.native_causal import FrequencyVariableDomain
from r0.modules.priors.lib.native_vector_intervention_execution import (
    forced_variable,
    vector_clamps,
)
from r0.modules.priors.lib.native_vector_records import NativeVectorNode, NativeVectorRecipe
from r0.modules.priors.lib.node_mechanisms import NodeMechanisms
from r0.modules.priors.lib.node_pipeline import PipelineEvaluation
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.signal_noise import signal_noise_scale
from r0.modules.priors.lib.source_root_base import functional_root_values

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class NativeVectorEvaluation:
    values: tuple[tuple[torch.Tensor, ...], ...]
    scores: tuple[torch.Tensor, ...]
    probabilities: dict[tuple[int, int], torch.Tensor]
    uniforms: dict[tuple[int, int], torch.Tensor]
    states: tuple[torch.Tensor, ...]


def native_vector_rng(seed: int, task_id: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(
        int(record_fingerprint(("native-vector-causal-v1", seed, task_id, keys)), 16)
    )


def evaluate_native_vector_graph(
    recipe: NativeVectorRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
    execution: InterventionExecution | None = None,
    mechanisms: NodeMechanisms | None = None,
) -> NativeVectorEvaluation:
    if type(recipe) is not NativeVectorRecipe or any(
        type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)
    ):
        raise ValueError("native vector evaluation requires frozen recipe and bounded identities")
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("native vector device unavailable")
    recipe.admit()
    return evaluate_native_vector_nodes(
        recipe.nodes,
        recipe.task.rows,
        lambda *keys: (
            execution.rng(lambda *k: native_vector_rng(seed, task_id, *k), *keys)
            if execution
            else native_vector_rng(seed, task_id, *keys)
        ),
        device,
        execution=execution,
        mechanisms=mechanisms,
        limits=recipe.limits,
    )


def evaluate_native_vector_nodes(
    nodes: tuple[NativeVectorNode, ...],
    rows: int,
    rng: Callable[..., np.random.Generator],
    device: str,
    *,
    execution: InterventionExecution | None = None,
    mechanisms: NodeMechanisms | None = None,
    capture: RawDrawLedger | None = None,
    owner: tuple[str, ...] = ("native",),
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> NativeVectorEvaluation:
    """Evaluate an already admitted prefix; callers own resource and certificate checks."""
    states: list[torch.Tensor] = []
    values: list[tuple[torch.Tensor, ...]] = []
    scores: list[torch.Tensor] = []
    probabilities: dict[tuple[int, int], torch.Tensor] = {}
    uniforms: dict[tuple[int, int], torch.Tensor] = {}
    for index, node in enumerate(nodes):
        equation = node.equation
        clamps = vector_clamps(execution, index)
        if len(clamps) == len(node.domains):
            forced = [
                forced_variable(clamps[j], node, rows, device) for j in range(len(node.domains))
            ]
            values.append(tuple(v[0] for v in forced))
            states.append(torch.cat([v[1] for v in forced], dim=1))
            scores.append(torch.cat([v[2] for v in forced], dim=1))
            if mechanisms:
                mechanisms.record(index, state=states[-1], scores=scores[-1])
                for j, (value, encoded, score) in enumerate(forced):
                    mechanisms.record(index, **{f"value:{j}": value, f"forced-score:{j}": score})
                    if node.domains[j].kind != "continuous":
                        mechanisms.record(index, **{f"probability:{j}": encoded})
            assert execution is not None
            execution.bypassed.append(index)
            execution.record(index, state=states[-1], score=scores[-1])
            for j, (value, encoded, score) in enumerate(forced):
                execution.record(index, **{f"value:{j}": value, f"forced-score:{j}": score})
                if node.domains[j].kind != "continuous":
                    probabilities[index, j] = encoded
                    execution.record(index, **{f"forced-probability:{j}": encoded})
            continue
        if equation.root:
            inputs = torch.tensor(
                functional_root_values(
                    equation.root,
                    rows,
                    rng("root", index),
                    capture=capture,
                    owner=(*owner, f"node:{index}", "root"),
                    limits=limits,
                ),
                dtype=torch.float64,
                device=device,
            )
        else:
            inputs = torch.cat([states[p] for p in equation.parents], dim=1)
        if execution:
            execution.record(index, inputs=inputs)
            if equation.root:
                execution.record(index, root=inputs)
        output = (
            mechanisms.evaluate(index, equation.function, inputs)
            if mechanisms
            else evaluate_admitted_function(equation.function, inputs)
        )
        raw = torch.tensor(
            sample_exogenous(
                equation.exogenous,
                rows,
                equation.function.output_width,
                rng,
                index,
                capture=capture,
                owner=(*owner, f"node:{index}", "innovation"),
            ),
            dtype=torch.float64,
            device=device,
        )
        scale = signal_noise_scale(
            output, equation.noise_scale or (0.0,) * output.shape[1], equation.noise_response
        )
        if execution:
            execution.record(index, function=output, exogenous=raw, scale=scale)
        pipeline = PipelineEvaluation(equation.pipeline, index, None) if equation.pipeline else None
        transformed = pipeline.before_noise(output) if pipeline else output
        state = transformed + scale * raw
        pre_finish = state
        if pipeline:
            state, _ = pipeline.finish(state)
        if not torch.isfinite(state).all():
            raise ValueError("nonfinite native equation score; task was not retried")
        scores.append(state)
        if mechanisms:
            mechanisms.record(
                index,
                inputs=inputs,
                function=output,
                exogenous=raw,
                scale=scale,
                before_noise=transformed,
                pre_finish=pre_finish,
                post_finish=state,
                scores=state,
            )
        joint = state
        parts: list[torch.Tensor] = []
        node_values: list[torch.Tensor] = []
        for variable, (domain, span) in enumerate(zip(node.domains, node.spans, strict=True)):
            state = joint[:, span.score_start : span.score_stop]
            if variable in clamps:
                value, encoded, forced_score = forced_variable(clamps[variable], node, rows, device)
                node_values.append(value)
                parts.append(encoded)
                joint[:, span.score_start : span.score_stop] = forced_score
                if mechanisms:
                    mechanisms.record(index, **{f"forced-score:{variable}": forced_score})
                assert execution is not None
                execution.record(
                    index, **{f"value:{variable}": value, f"forced-score:{variable}": forced_score}
                )
                if domain.kind != "continuous":
                    probabilities[index, variable] = encoded
                    execution.record(index, **{f"forced-probability:{variable}": encoded})
                continue
            if domain.kind == "continuous":
                node_values.append(state[:, 0])
                parts.append(state)
                continue
            count = len(domain.vocabulary)
            emission_scores = state
            if isinstance(domain, FrequencyVariableDomain) and domain.emission != "threshold":
                log_weights = torch.tensor(
                    domain.frequencies, dtype=torch.float64, device=device
                ).log()
                emission_scores = (
                    state - state.max(dim=1, keepdim=True).values
                ) / domain.temperature + log_weights
            if domain.emission == "threshold":
                cuts = torch.tensor(domain.thresholds, dtype=torch.float64, device=device)
                category = torch.bucketize(state[:, 0].contiguous(), cuts, right=True)
            elif domain.emission == "argmax":
                category = torch.argmax(emission_scores, dim=1)
            else:
                temperature = (
                    1.0 if isinstance(domain, FrequencyVariableDomain) else domain.temperature
                )
                probability = torch.softmax(
                    (emission_scores - emission_scores.max(dim=1, keepdim=True).values)
                    / temperature,
                    dim=1,
                )
                if not torch.isfinite(probability).all():
                    raise ValueError("nonfinite native emission probability; task was not retried")
                uniform = torch.tensor(
                    recorded_draw(
                        rng("native-emission", index, variable),
                        capture,
                        owner=(*owner, f"node:{index}", f"variable:{variable}", "emission"),
                        role="uniform",
                        distribution="random",
                        shape=(rows,),
                    ),
                    dtype=torch.float64,
                    device=device,
                )
                # Force the final CDF endpoint to one to avoid a rounding-created extra category.
                cumulative = probability.cumsum(dim=1)
                cumulative[:, -1] = 1.0
                category = (uniform[:, None] >= cumulative).sum(dim=1)
                probabilities[index, variable], uniforms[index, variable] = probability, uniform
                if execution:
                    execution.record(index, **{f"uniform:{variable}": uniform})
            encoded = torch.nn.functional.one_hot(category, num_classes=count).to(torch.float64)
            probabilities.setdefault((index, variable), encoded)
            node_values.append(category)
            parts.append(encoded)
        values.append(tuple(node_values))
        states.append(torch.cat(parts, dim=1))
        if mechanisms:
            mechanisms.record(index, state=states[-1], scores=joint)
            for j, value in enumerate(node_values):
                mechanisms.record(index, **{f"value:{j}": value})
                if (index, j) in probabilities:
                    mechanisms.record(index, **{f"probability:{j}": probabilities[index, j]})
                if (index, j) in uniforms:
                    mechanisms.record(index, **{f"uniform:{j}": uniforms[index, j]})
        if execution:
            execution.record(index, state=states[-1], score=joint)
            for j, value in enumerate(node_values):
                execution.record(index, **{f"value:{j}": value})
    return NativeVectorEvaluation(
        tuple(values), tuple(scores), probabilities, uniforms, tuple(states)
    )
