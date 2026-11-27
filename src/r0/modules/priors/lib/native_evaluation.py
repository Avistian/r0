"""Evaluate typed DAG states and native targets without a post-hoc label adapter."""

import platform
from dataclasses import asdict, dataclass

import numpy as np
import torch

from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.joint_root_base import draw_root_base
from r0.modules.priors.lib.native_causal import NativeCausalRecipe
from r0.modules.priors.lib.node_pipeline import PipelineEvaluation
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.signal_noise import signal_noise_scale
from r0.modules.priors.lib.training_features import TrainingFeature
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class NativeEvaluation:
    values: tuple[torch.Tensor, ...]
    scores: tuple[torch.Tensor, ...]
    probabilities: dict[int, torch.Tensor]
    uniforms: dict[int, torch.Tensor]


def native_rng(seed: int, task_id: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(
        int(record_fingerprint(("native-causal-v1", seed, task_id, keys)), 16)
    )


def evaluate_native_graph(
    recipe: NativeCausalRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> NativeEvaluation:
    """Discrete node values are int64; their descendants receive explicit one-hot state."""
    if type(recipe) is not NativeCausalRecipe:
        raise ValueError("native evaluation requires its validated recipe")
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("native seed and task identity must be bounded nonnegative integers")
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("native device must be available cpu or cuda; no implicit fallback")
    recipe.admit()
    rows = recipe.task.rows
    states: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    probabilities: dict[int, torch.Tensor] = {}
    uniforms: dict[int, torch.Tensor] = {}

    def rng(*keys: object) -> np.random.Generator:
        return native_rng(seed, task_id, *keys)

    for index, node in enumerate(recipe.nodes):
        equation, domain = node.equation, node.domain
        if equation.root:
            inputs = torch.tensor(
                draw_root_base(equation.root, rows, rng("root", index)).values,
                dtype=torch.float64,
                device=device,
            )
        else:
            inputs = torch.cat([states[p] for p in equation.parents], dim=1)
        output = evaluate_admitted_function(equation.function, inputs)
        raw = torch.tensor(
            sample_exogenous(equation.exogenous, rows, equation.function.output_width, rng, index),
            dtype=torch.float64,
            device=device,
        )
        scale = signal_noise_scale(
            output, equation.noise_scale or (0.0,) * output.shape[1], equation.noise_response
        )
        pipeline = PipelineEvaluation(equation.pipeline, index, None) if equation.pipeline else None
        state = (pipeline.before_noise(output) if pipeline else output) + scale * raw
        if pipeline:
            state, _ = pipeline.finish(state)
        if not torch.isfinite(state).all():
            raise ValueError("nonfinite native equation score; task was not retried")
        scores.append(state)
        if domain.kind == "continuous":
            values.append(state)
            states.append(state)
            continue
        count = len(domain.vocabulary)
        if domain.emission == "threshold":
            cuts = torch.tensor(domain.thresholds, dtype=torch.float64, device=device)
            category = torch.bucketize(state[:, 0].contiguous(), cuts, right=True)
        elif domain.emission == "argmax":
            category = torch.argmax(state, dim=1)
        else:
            probability = torch.softmax(
                (state - state.max(dim=1, keepdim=True).values) / domain.temperature, dim=1
            )
            if not torch.isfinite(probability).all():
                raise ValueError("nonfinite native emission probability; task was not retried")
            uniform = torch.tensor(
                rng("native-emission", index).random(rows), dtype=torch.float64, device=device
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
    return NativeEvaluation(tuple(values), tuple(scores), probabilities, uniforms)


def generate_native_causal(
    recipe: NativeCausalRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    result = evaluate_native_graph(recipe, seed=seed, task_id=task_id, device=device)
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
