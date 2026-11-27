"""One admitted evaluator for frozen DAG generation and original-row inspection."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.functional_analysis import functional_certificates
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_causal_decoding import functional_recipe_from_dict
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.functional_target_calibration import (
    fit_target_calibration,
    target_calibration_from_dict,
)
from r0.modules.priors.lib.joint_root_base import draw_root_base
from r0.modules.priors.lib.joint_sampling_records import validate_path
from r0.modules.priors.lib.node_pipeline import PipelineEvaluation
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.signal_noise import signal_noise_scale
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


def functional_rng(seed: int, task_id: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(
        int(record_fingerprint(("functional-causal-v1", seed, task_id, keys)), 16)
    )


def _constant_diagnostic(
    value: torch.Tensor, node: int, stage: str, diagnostics: list[dict[str, Any]] | None
) -> None:
    if diagnostics is None:
        return
    constant: list[int] = (
        torch.all(value == value[0], dim=0).nonzero().flatten().detach().cpu().numpy().tolist()
    )
    if constant:
        diagnostics.append(
            {
                "node": node,
                "stage": stage,
                "rows": len(value),
                "constant_channels": constant,
                "flags": [f"constant_{stage}_sample"],
                "meaning": "constant in these emitted rows; not a population certificate",
            }
        )


def functional_values(
    recipe: FunctionalCausalRecipe,
    rng: Callable[..., np.random.Generator],
    device: str,
    *,
    add: Callable[..., None] | None = None,
    indices: Sequence[int] = (),
    readouts: list[torch.Tensor] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    rows: int | None = None,
) -> list[torch.Tensor]:
    rows = recipe.task.rows if rows is None else rows
    recipe.admit(evaluation_rows=rows)
    return _evaluate_functional_nodes(
        recipe.nodes,
        rows,
        rng,
        device,
        add=add,
        indices=indices,
        readouts=readouts,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True, slots=True)
class PrefixEvaluation:
    states: tuple[torch.Tensor, ...]
    readouts: tuple[torch.Tensor, ...]


def evaluate_functional_prefix(
    prefix: FunctionalPrefix, *, rng: Callable[..., np.random.Generator], device: str = "cpu"
) -> PrefixEvaluation:
    """Evaluate admitted ancestors; the builder supplies fresh streams for each receiving fit."""
    if type(prefix) is not FunctionalPrefix:
        raise ValueError("functional prefix evaluation requires its admitted frozen record")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; no implicit fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select CPU explicitly")
    prefix.admit()
    readouts: list[torch.Tensor] = []
    states = _evaluate_functional_nodes(prefix.nodes, prefix.rows, rng, device, readouts=readouts)
    return PrefixEvaluation(tuple(states), tuple(readouts))


def sample_functional_prefix(
    prefix: FunctionalPrefix,
    *,
    rng: np.random.Generator,
    owner: tuple[str, ...],
    capture: RawDrawLedger | None = None,
    device: str = "cpu",
) -> PrefixEvaluation:
    """Draw a fresh ancestor population atomically from one calibration stream.

    All roots and innovations consume this stream in node/column/draw order.
    The keyed evaluator remains available for authored generation and inspection.
    """
    if type(prefix) is not FunctionalPrefix:
        raise ValueError("functional prefix sampling requires its admitted frozen record")
    validate_path(owner)
    if len(owner) > 13:
        raise ValueError("prefix sampling owner must leave room for node and column paths")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; no implicit fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select CPU explicitly")
    prefix.admit()
    with raw_transactions({"calibration": rng}, capture) as (streams, ledger):
        readouts: list[torch.Tensor] = []
        states = _evaluate_functional_nodes(
            prefix.nodes,
            prefix.rows,
            lambda *keys: streams["calibration"],
            device,
            readouts=readouts,
            capture=ledger,
            owner=owner,
        )
        return PrefixEvaluation(tuple(states), tuple(readouts))


def _evaluate_functional_nodes(
    nodes: tuple[FunctionalCausalNode, ...],
    rows: int,
    rng: Callable[..., np.random.Generator],
    device: str,
    *,
    add: Callable[..., None] | None = None,
    indices: Sequence[int] = (),
    readouts: list[torch.Tensor] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    capture: RawDrawLedger | None = None,
    owner: tuple[str, ...] = (),
) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    for index, node in enumerate(nodes):
        parents = [f"causal-{p}" for p in node.parents]
        if node.root is not None:
            sample = draw_root_base(
                node.root,
                rows,
                rng("root-base", index),
                capture=capture,
                owner=(*owner, f"node:{index}", "root"),
            )
            inputs = torch.tensor(sample.values, dtype=torch.float64, device=device)
            parents = [f"root-{index}"]
            if add is not None:
                add(
                    parents[0],
                    f"Root base {index}",
                    [],
                    "frozen base law applied to raw draws",
                    inputs[list(indices)],
                    asdict(node.root),
                )
        else:
            inputs = torch.cat([values[p] for p in node.parents], dim=1)
        function_diagnostics: list[dict[str, Any]] | None = [] if diagnostics is not None else None
        output = evaluate_admitted_function(node.function, inputs, diagnostics=function_diagnostics)
        if diagnostics is not None:
            diagnostics.extend({"node": index, **entry} for entry in function_diagnostics or ())
        _constant_diagnostic(output, index, "function", diagnostics)
        raw = torch.tensor(
            sample_exogenous(
                node.exogenous,
                rows,
                node.function.output_width,
                rng,
                index,
                capture=capture,
                owner=(*owner, f"node:{index}", "innovation"),
            ),
            dtype=torch.float64,
            device=device,
        )
        scale = signal_noise_scale(
            output, node.noise_scale or (0.0,) * node.function.output_width, node.noise_response
        )
        pipeline = (
            PipelineEvaluation(node.pipeline, index, diagnostics, record_steps=add is not None)
            if node.pipeline
            else None
        )
        transformed = pipeline.before_noise(output) if pipeline else output
        state = transformed + scale * raw
        state, readout = pipeline.finish(state) if pipeline else (state, state)
        if not torch.isfinite(state).all() or not torch.isfinite(readout).all():
            raise ValueError("nonfinite functional causal value; task was not retried")
        _constant_diagnostic(state, index, "state", diagnostics)
        values.append(state)
        if readouts is not None:
            readouts.append(readout)
        if add is not None:
            add(
                f"function-{index}",
                f"Frozen function {index}",
                parents,
                "f(concat(parent states))" if node.parents else "f(root base)",
                output[list(indices)],
                asdict(node.function),
            )
            add(
                f"exogenous-{index}",
                f"Structural exogenous draw {index}",
                [],
                "draw from frozen structural law",
                raw[list(indices)],
                asdict(node.exogenous),
            )
            add(
                f"noise-scale-{index}",
                f"Signal-dependent scale {index}",
                [f"function-{index}"],
                "noise_scale * (offset + strength * abs(function output)**power)",
                scale[list(indices)],
                asdict(node.noise_response),
            )
            state_parents = [f"function-{index}", f"exogenous-{index}", f"noise-scale-{index}"]
            readout_parents = [f"causal-{index}"]
            if pipeline is not None:
                previous = [f"function-{index}"]
                for name, value in pipeline.steps:
                    key = f"pipeline-{index}-{name}"
                    add(
                        key,
                        f"Pipeline {index}: {name}",
                        previous,
                        "frozen pipeline stage before structural innovation and node importance",
                        value[list(indices)],
                        {"stage": name},
                    )
                    if name not in ("converter-state", "converted-readout"):
                        previous = [key]
                state_parents[0] = f"pipeline-{index}-converter-state"
                count = len(pipeline.pipeline.converters)
                readout_parents = ([f"pipeline-{index}-converted-readout"] if count else []) + (
                    [f"causal-{index}"] if count < node.function.output_width else []
                )
            add(
                f"causal-{index}",
                f"Causal state {index}",
                state_parents,
                "node_importance * (pipeline(function output) + scale * exogenous)",
                state[list(indices)],
                {"pipeline": asdict(node.pipeline) if node.pipeline else None},
            )
            add(
                f"readout-{index}",
                f"Readout {index}",
                readout_parents,
                "captured converter channels before innovation/importance; otherwise final state",
                readout[list(indices)],
                {},
            )
    return values


def generate_functional_causal(
    recipe: FunctionalCausalRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; no implicit fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select CPU explicitly")

    def rng(*keys: object) -> np.random.Generator:
        return functional_rng(seed, task_id, *keys)

    values: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    functional_values(recipe, rng, device, readouts=values, diagnostics=diagnostics)
    inputs = torch.stack([values[i][:, j] for i, j in recipe.observed], dim=1)
    scores = torch.stack([values[i][:, j] for i, j in recipe.target], dim=1)
    calibration = None
    if recipe.target_calibration_rows:
        independent: list[torch.Tensor] = []
        functional_values(
            recipe,
            lambda *keys: rng("target-calibration-v1", *keys),
            "cpu",
            readouts=independent,
            rows=recipe.target_calibration_rows,
        )
        raw = torch.stack([independent[i][:, j] for i, j in recipe.target], dim=1)
        calibration = fit_target_calibration(
            raw, record_fingerprint(([asdict(n) for n in recipe.nodes], recipe.target))
        )
        scores = calibration.apply(scores)
    _, certificates = functional_certificates(recipe.nodes)
    target_sources = frozenset[tuple[int, int]]().union(
        *(certificates[i][j].sources for i, j in recipe.target)
    )
    active = [
        column
        for column, (i, j) in enumerate(recipe.observed)
        if certificates[i][j].sources & target_sources
    ]
    flags = [] if active else ["disconnected_target"]
    if torch.all(scores == scores[0]).item():
        flags.append("zero_target_dispersion")
    flags.extend(sorted({flag for entry in diagnostics for flag in entry["flags"]}))
    card = {
        "version": recipe.version,
        "family": "causal",
        "construction": "authored-frozen-functions-v1",
        "nodes": [asdict(n) for n in recipe.nodes],
        "pipelines": [asdict(n.pipeline) if n.pipeline else None for n in recipe.nodes],
        "observed": recipe.observed,
        "target": recipe.target,
        "active": active,
        "active_definition": (
            "conservative channel-level exogenous ancestry; potential dependence only"
        ),
        "flags": flags,
        "node_diagnostics": diagnostics,
        "normalization": "none",
        "moment_contract": {
            "target_orders": [
                "all" if certificates[i][j].order == float("inf") else certificates[i][j].order
                for i, j in recipe.target
            ],
            "meaning": (
                "positive absolute moments strictly below each bound are certified finite; "
                "zero means unknown"
            ),
            "normalization_requires": (
                "certified finite second moments, including internal MLP normalization"
            ),
        },
        "seed": seed,
        "task_id": task_id,
        "resource_limits": asdict(recipe.limits),
        "resource_admission": recipe.admit(),
        "innovation": "recorded exogenous laws; independently keyed root and structural streams",
        "sampling_provenance": "authored",
    }
    if calibration is not None:
        card["normalization"] = "fixed independent calibration"
        card["calibration"] = asdict(calibration)
    return complete_task(
        recipe.task, asdict(recipe), card, inputs, scores, seed=seed, task_id=task_id, rng=rng
    )


def functional_steps(
    record: dict[str, Any], card: dict[str, Any], indices: Sequence[int], add: Callable[..., None]
) -> torch.Tensor:
    recipe = functional_recipe_from_dict(record["recipe"])
    values: list[torch.Tensor] = []
    functional_values(
        recipe,
        lambda *keys: functional_rng(record["seed"], record["task_id"], *keys),
        "cpu",
        add=add,
        indices=indices,
        readouts=values,
    )
    scores = torch.stack([values[i][list(indices), j] for i, j in recipe.target], dim=1)
    parameters: dict[str, Any] = {"target": recipe.target}
    if recipe.target_calibration_rows:
        calibration = target_calibration_from_dict(card.get("calibration"), len(recipe.target))
        if (
            calibration.rows != recipe.target_calibration_rows
            or calibration.prefix_fingerprint
            != record_fingerprint(([asdict(n) for n in recipe.nodes], recipe.target))
        ):
            raise ValueError("retained target calibration differs from its frozen graph")
        scores = calibration.apply(scores)
        parameters["calibration"] = card["calibration"]
    add(
        "signal",
        "Graph target channels",
        list(dict.fromkeys(f"readout-{i}" for i, _ in recipe.target)),
        "(declared target channels - fixed center) / fixed scale"
        if recipe.target_calibration_rows
        else "select declared target channels",
        scores,
        parameters,
    )
    return scores
