"""Evaluate typed DAG states and native targets without a post-hoc label adapter."""

import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.native_causal import (
    FrequencyVariableDomain,
    NativeCausalNode,
    NativeCausalRecipe,
)
from r0.modules.priors.lib.node_mechanisms import NodeMechanisms
from r0.modules.priors.lib.node_pipeline import PipelineEvaluation
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, recorded_draw
from r0.modules.priors.lib.recursive_function_metadata import recursive_mechanism_metadata
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.root_populations import RootPopulations
from r0.modules.priors.lib.signal_noise import signal_noise_scale
from r0.modules.priors.lib.source_root_base import functional_root_values
from r0.modules.priors.lib.training_features import TrainingFeature
from r0.modules.priors.lib.training_task import TrainingTask

_DEFAULT_LIMITS = CausalLimits()


@dataclass(frozen=True, slots=True)
class NativeEvaluation:
    values: tuple[torch.Tensor, ...]
    scores: tuple[torch.Tensor, ...]
    probabilities: dict[int, torch.Tensor]
    uniforms: dict[int, torch.Tensor]
    states: tuple[torch.Tensor, ...]


def native_rng(seed: int, task_id: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(
        int(record_fingerprint(("native-causal-v1", seed, task_id, keys)), 16)
    )


def evaluate_native_graph(
    recipe: NativeCausalRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
    execution: InterventionExecution | None = None,
    mechanisms: NodeMechanisms | None = None,
    root_populations: RootPopulations | None = None,
) -> NativeEvaluation:
    """Discrete node values are int64; their descendants receive explicit one-hot state."""
    if type(recipe) is not NativeCausalRecipe:
        raise ValueError("native evaluation requires its validated recipe")
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("native seed and task identity must be bounded nonnegative integers")
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("native device must be available cpu or cuda; no implicit fallback")
    recipe.admit()
    return evaluate_native_nodes(
        recipe.nodes,
        recipe.task.rows,
        lambda *keys: (
            execution.rng(lambda *parts: native_rng(seed, task_id, *parts), *keys)
            if execution
            else native_rng(seed, task_id, *keys)
        ),
        device,
        execution=execution,
        mechanisms=mechanisms,
        root_populations=root_populations,
        limits=recipe.limits,
    )


def evaluate_native_nodes(
    nodes: tuple[NativeCausalNode, ...],
    rows: int,
    rng: Callable[..., np.random.Generator],
    device: str,
    *,
    capture: RawDrawLedger | None = None,
    owner: tuple[str, ...] = ("native",),
    execution: InterventionExecution | None = None,
    mechanisms: NodeMechanisms | None = None,
    root_populations: RootPopulations | None = None,
    limits: CausalLimits = _DEFAULT_LIMITS,
) -> NativeEvaluation:
    """Evaluate an already admitted prefix; callers own resource and certificate checks."""
    states: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    probabilities: dict[int, torch.Tensor] = {}
    uniforms: dict[int, torch.Tensor] = {}
    for index, node in enumerate(nodes):
        equation, domain = node.equation, node.domain
        forced = execution.category(index) if execution else None
        if forced is not None:
            category = torch.full(
                (rows,), domain.vocabulary.index(forced), dtype=torch.int64, device=device
            )
            encoded = torch.nn.functional.one_hot(category, num_classes=len(domain.vocabulary)).to(
                torch.float64
            )
            values.append(category[:, None])
            states.append(encoded)
            scores.append(encoded)
            probabilities[index] = encoded
            if mechanisms:
                mechanisms.record(
                    index, state=encoded, readout=category[:, None], probability=encoded
                )
            assert execution is not None
            execution.bypassed.append(index)
            execution.record(
                index, state=encoded, readout=category[:, None], forced_probability=encoded
            )
            continue
        bypass = (
            execution.bypass(index, equation, rows, device, capture_readout=False)
            if execution and domain.kind == "continuous"
            else None
        )
        if bypass is not None:
            state, _ = bypass
            if mechanisms:
                mechanisms.record(index, state=state, readout=state)
            states.append(state)
            values.append(state)
            scores.append(state)
            continue
        if equation.root:
            external = (
                root_populations.values(index, rows, equation.function.input_width, device)
                if root_populations
                else None
            )
            if external is not None:
                if capture is not None:
                    raise ValueError("external root populations cannot create iid draw receipts")
                inputs = external
            else:
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
            if execution:
                execution.record(index, root=inputs)
        else:
            inputs = torch.cat([states[p] for p in equation.parents], dim=1)
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
        pipeline = PipelineEvaluation(equation.pipeline, index, None) if equation.pipeline else None
        transformed = pipeline.before_noise(output) if pipeline else output
        state = transformed + scale * raw
        pre_finish = state
        if pipeline:
            state, _ = pipeline.finish(state)
        if mechanisms:
            mechanisms.record(index, post_finish=state)
        if execution:
            execution.record(index, inputs=inputs, function=output, exogenous=raw, scale=scale)
            if domain.kind == "continuous":
                state, _ = execution.apply(index, equation, state, state)
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
                scores=state,
            )
        if domain.kind == "continuous":
            values.append(state)
            states.append(state)
            if mechanisms:
                mechanisms.record(index, state=state, readout=state)
            continue
        count = len(domain.vocabulary)
        emission_scores = state
        if isinstance(domain, FrequencyVariableDomain) and domain.emission != "threshold":
            log_weights = torch.tensor(domain.frequencies, dtype=torch.float64, device=device).log()
            emission_scores = (
                state - state.max(dim=1, keepdim=True).values
            ) / domain.temperature + log_weights
        if domain.emission == "threshold":
            cuts = torch.tensor(domain.thresholds, dtype=torch.float64, device=device)
            category = torch.bucketize(state[:, 0].contiguous(), cuts, right=True)
        elif domain.emission == "argmax":
            category = torch.argmax(emission_scores, dim=1)
        else:
            temperature = 1.0 if isinstance(domain, FrequencyVariableDomain) else domain.temperature
            probability = torch.softmax(
                (emission_scores - emission_scores.max(dim=1, keepdim=True).values) / temperature,
                dim=1,
            )
            if not torch.isfinite(probability).all():
                raise ValueError("nonfinite native emission probability; task was not retried")
            uniform = torch.tensor(
                recorded_draw(
                    rng("native-emission", index),
                    capture,
                    owner=(*owner, f"node:{index}", "emission"),
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
            probabilities[index], uniforms[index] = probability, uniform
        encoded = torch.nn.functional.one_hot(category, num_classes=count).to(torch.float64)
        probabilities.setdefault(index, encoded)
        values.append(category[:, None])
        states.append(encoded)
        if mechanisms:
            mechanisms.record(
                index, state=encoded, readout=category[:, None], probability=probabilities[index]
            )
            if index in uniforms:
                mechanisms.record(index, uniform=uniforms[index])
        if execution:
            execution.record(
                index, state=encoded, readout=category[:, None], probability=probabilities[index]
            )
            if index in uniforms:
                execution.record(index, uniform=uniforms[index])
    return NativeEvaluation(tuple(values), tuple(scores), probabilities, uniforms, tuple(states))


def generate_native_causal(
    recipe: NativeCausalRecipe,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
    execution: InterventionExecution | None = None,
    mechanisms: NodeMechanisms | None = None,
    root_populations: RootPopulations | None = None,
) -> TrainingTask:
    result = evaluate_native_graph(
        recipe,
        seed=seed,
        task_id=task_id,
        device=device,
        execution=execution,
        mechanisms=mechanisms,
        root_populations=root_populations,
    )
    config = asdict(recipe)
    source = {
        **config,
        "task": {
            k: v for k, v in asdict(recipe.task).items() if k not in ("context_rows", "split")
        },
    }
    source_id = record_fingerprint((source, seed, task_id))
    target_node, target_channel = recipe.target
    domain = recipe.nodes[target_node].domain
    inputs = torch.stack(
        [result.values[i][:, j].to(torch.float64) for i, j in recipe.observed], dim=1
    )
    targets = result.values[target_node][:, target_channel]
    permutation = torch.tensor(
        native_rng(seed, task_id, "split", recipe.task.split).permutation(recipe.task.rows),
        device=device,
    )
    emissions = {
        str(i): {
            "domain": asdict(node.domain),
            "uniforms": result.uniforms[i].detach().cpu().numpy().tolist()
            if i in result.uniforms
            else None,
        }
        for i, node in enumerate(recipe.nodes)
        if node.domain.kind != "continuous"
    }
    return TrainingTask(
        source_id,
        record_fingerprint((source_id, recipe.task.split, recipe.task.context_rows)),
        recipe.task.task_type,
        encode(
            {
                "recipe": config,
                "seed": seed,
                "task_id": task_id,
                "execution": {
                    "implementation": recipe.version,
                    "device": device,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": str(torch.__version__),
                    "cpu_threads": torch.get_num_threads(),
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                },
            }
        ),
        encode(
            {
                "family": "native_causal",
                **recursive_mechanism_metadata(tuple(n.equation.function for n in recipe.nodes)),
                "version": recipe.version,
                "target_origin": "native-variable-v1",
                "target": recipe.target,
                "native_emissions": emissions,
                "admission": recipe.admit(),
            }
        ),
        inputs,
        targets,
        targets[:, None] if domain.kind == "continuous" else result.scores[target_node],
        result.probabilities.get(target_node),
        permutation[: recipe.task.context_rows],
        permutation[recipe.task.context_rows :],
        domain.vocabulary,
        tuple(
            TrainingFeature(recipe.nodes[i].domain.kind, recipe.nodes[i].domain.vocabulary)
            for i, _ in recipe.observed
        ),
        "native-variable-v1",
    )
