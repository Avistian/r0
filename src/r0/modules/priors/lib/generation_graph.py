"""Reconstruct a bounded generation graph from saved parameters and random streams."""

import json
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import torch

from r0.modules.priors.lib.causal import causal_rng
from r0.modules.priors.lib.causal_inspection import causal_steps
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.functional_steps import functional_steps
from r0.modules.priors.lib.graph_functions import analytic_steps, composite_steps, gp_steps
from r0.modules.priors.lib.native_inspection import native_steps
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.signals import component_inputs, multiply_signals
from r0.modules.priors.lib.training_task import TrainingTask


def _preview(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _preview(item) for key, item in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        items = cast(list[Any], value)
        shown = [_preview(item) for item in items[:32]]
        return shown if len(items) <= 32 else {"total_items": len(items), "shown": shown}
    return value


def explain_generation(
    task: TrainingTask,
    indices: Sequence[int],
    *,
    component: int | None = None,
    column_start: int = 0,
    limits: CausalLimits | None = None,
) -> dict[str, Any]:
    """Return selectable graph nodes for 1..16 original rows, without altering the task.

    Expand a composite's child with ``component``. Values show up to 32 columns
    starting at ``column_start`` (clamped for narrower nodes), with explicit indices
    and full shapes. Parameter lists show their first 32 items at each level; full
    parameters remain in mechanism_json. For frozen functional graphs, pass caller-owned
    limits to authorize exploration caps above the defaults. GP nodes reconstruct the original joint
    draw, not an interpolation or a newly sampled extension. All work runs on CPU.
    """
    if limits is not None and type(limits) is not CausalLimits:
        raise ValueError("inspection requires caller-owned causal limits")
    if not 1 <= len(indices) <= 16:
        raise ValueError("select 1..16 original rows")
    if any(type(i) is not int or not 0 <= i < len(task.inputs) for i in indices):
        raise ValueError("invalid original row index")
    if type(column_start) is not int or not 0 <= column_start <= 4095:
        raise ValueError("column_start must be 0..4095")
    record = json.loads(task.recipe_json)
    card = json.loads(task.mechanism_json)
    inputs = task.inputs.detach().cpu()[list(indices)].clone()
    nodes: list[dict[str, Any]] = []

    def add(
        key: str,
        label: str,
        parents: list[str],
        formula: str,
        values: torch.Tensor,
        parameters: dict[str, Any],
    ) -> None:
        matrix = values.reshape(len(values), -1).detach().cpu().numpy()
        if not np.isfinite(matrix).all():
            raise ValueError(f"nonfinite graph values at {key}; no resampling or replacement")
        start = min(column_start, matrix.shape[1] - 1)
        end = min(start + 32, matrix.shape[1])
        nodes.append(
            {
                "id": key,
                "label": label,
                "parents": parents,
                "formula": formula,
                "shape": list(matrix.shape),
                "column_indices": list(range(start, end)),
                "values": np.asarray(matrix[:, start:end], dtype=np.float64).tolist(),
                "parameters": _preview(parameters),
            }
        )

    if record["recipe"].get("version") in ("native-causal-v1", "sampled-native-graph-v1"):
        if component is not None:
            raise ValueError("native graphs do not have composite children")
        return {**native_steps(task, indices, add, limits=limits), "nodes": nodes}

    components = card.get("components", [])
    if component is not None:
        if type(component) is not int or not 0 <= component < len(components):
            raise ValueError("component must identify a saved composition child")
        child = components[component]
        mapped = component_inputs(child, inputs)
        add(
            "inputs",
            "Mapped component inputs",
            [],
            "child active columns = parent columns",
            mapped,
            {
                "parent_columns": child["input_columns"],
                "child_columns": child["mechanism"]["active"],
                "seed": child["seed"],
                "inactive_columns": "zero; recorded child weights ignore them",
            },
        )
        signal = analytic_steps(child["mechanism"], mapped, add)
        add(
            "component",
            "Component strength",
            ["signal"],
            "strength * normalized child signal",
            multiply_signals(signal, child["strength"]),
            {"strength": child["strength"]},
        )
        return {
            "version": "generation-graph-v1",
            "episode_id": task.episode_id,
            "rows": list(indices),
            "component": component,
            "component_count": len(components),
            "nodes": nodes,
        }

    add(
        "inputs",
        "Observed inputs",
        list(
            dict.fromkeys(
                f"readout-{i}" if "pipelines" in card else f"causal-{i}"
                for i, _ in card["observed"]
            )
        )
        if card["version"]
        in (
            "causal-v1",
            "causal-v2",
            "functional-causal-v1",
            "fitted-functional-causal-v1",
            "functional-causal-v2",
            "sampled-functional-graph-v1",
            "sampled-parent-graph-v1",
            "sampled-topology-graph-v1",
            "sampled-geometry-graph-v1",
        )
        else [],
        "select declared converter readouts"
        if "pipelines" in card
        else "select declared causal node channels"
        if card["version"]
        in (
            "causal-v1",
            "causal-v2",
            "functional-causal-v1",
            "fitted-functional-causal-v1",
            "functional-causal-v2",
            "sampled-functional-graph-v1",
            "sampled-parent-graph-v1",
            "sampled-topology-graph-v1",
            "sampled-geometry-graph-v1",
        )
        else "independent per-column input streams",
        inputs,
        {
            "distribution": card.get("inputs", card.get("input")),
            "active_columns": card["active"],
            "row_indices": list(indices),
            **(
                {"observed_channels": card["observed"]}
                if card["version"]
                in (
                    "causal-v1",
                    "causal-v2",
                    "functional-causal-v1",
                    "fitted-functional-causal-v1",
                    "functional-causal-v2",
                    "sampled-functional-graph-v1",
                    "sampled-parent-graph-v1",
                    "sampled-topology-graph-v1",
                    "sampled-geometry-graph-v1",
                )
                else {}
            ),
        },
    )
    causal = card["version"] in (
        "causal-v1",
        "causal-v2",
        "functional-causal-v1",
        "fitted-functional-causal-v1",
        "functional-causal-v2",
        "sampled-functional-graph-v1",
        "sampled-parent-graph-v1",
        "sampled-topology-graph-v1",
        "sampled-geometry-graph-v1",
    )
    observed_step = nodes.pop(0) if causal else None
    if card["version"] in (
        "functional-causal-v1",
        "fitted-functional-causal-v1",
        "functional-causal-v2",
        "sampled-functional-graph-v1",
        "sampled-parent-graph-v1",
        "sampled-topology-graph-v1",
        "sampled-geometry-graph-v1",
    ):
        signal = functional_steps(record, card, indices, add, limits=limits)
    elif card["version"] in ("causal-v1", "causal-v2"):
        signal = causal_steps(card, len(task.inputs), indices, add)
    elif card["version"] == "exact-gp-v1":
        signal = gp_steps(card, record["recipe"], indices, add)
    elif card["version"] == "composite-v1":
        signal = composite_steps(card, inputs, add)
        for i in range(len(components)):
            next(node for node in nodes if node["id"] == f"component-{i}")["component"] = i
    else:
        signal = analytic_steps(card, inputs, add)
    if observed_step is not None:
        # The renderer lays out nodes in emission order. Observations depend on
        # readouts produced inside the causal graph, not the other way round.
        nodes.append(observed_step)
    expected = task.scores.detach().cpu()[list(indices)]
    if not torch.allclose(signal, expected, rtol=1e-10, atol=1e-10):
        raise ValueError("reconstructed signal differs from saved values beyond 1e-10 tolerance")
    controls = (
        record["recipe"]["request"]["task"]
        if card["version"]
        in (
            "sampled-functional-graph-v1",
            "sampled-parent-graph-v1",
            "sampled-topology-graph-v1",
            "sampled-geometry-graph-v1",
        )
        else record["recipe"].get("task", record["recipe"])
    )

    def rng(stream: str) -> np.random.Generator:
        recipe = record["recipe"]
        if recipe["version"] in (
            "functional-causal-v1",
            "fitted-functional-causal-v1",
            "functional-causal-v2",
            "sampled-functional-graph-v1",
            "sampled-parent-graph-v1",
            "sampled-topology-graph-v1",
            "sampled-geometry-graph-v1",
        ):
            return functional_rng(record["seed"], record["task_id"], stream)
        if recipe["version"] in ("causal-v1", "causal-v2"):
            return causal_rng(record["seed"], record["task_id"], stream)
        namespace: tuple[object, ...] = (recipe["version"],)
        if recipe["version"] in ("fourier-v1", "smooth-v1"):
            namespace += (recipe["family"],)
        if recipe["version"] == "smooth-v1":
            namespace += (recipe["projection"],)
        identity = (*namespace, record["seed"], record["task_id"], (stream,))
        return np.random.default_rng(int(record_fingerprint(identity), 16))

    targets = _target_steps(controls, signal, indices, rng, add)
    original_targets = task.targets.detach().cpu()[list(indices)]
    if not torch.allclose(targets, original_targets, rtol=1e-10, atol=1e-10):
        raise ValueError("reconstructed targets differ from saved values beyond 1e-10 tolerance")
    context = set(task.context_indices.detach().cpu().numpy().tolist())
    add(
        "split",
        "Context / query assignment",
        ["inputs", "target"],
        "independent split permutation; 0 = context, 1 = query",
        torch.tensor([0 if i in context else 1 for i in indices]),
        {
            "split": controls["split"],
            "context_rows": controls["context_rows"],
            "row_indices": list(indices),
        },
    )
    return {
        "version": "generation-graph-v1",
        "episode_id": task.episode_id,
        "rows": list(indices),
        "component": None,
        "component_count": len(components),
        "nodes": nodes,
        "signal_max_error": float((signal - expected).abs().max()),
        "target_max_error": float((targets - original_targets).abs().max()),
    }


def _target_steps(
    controls: dict[str, Any],
    signal: torch.Tensor,
    indices: Sequence[int],
    rng: Callable[[str], np.random.Generator],
    add: Callable[..., None],
) -> torch.Tensor:
    if controls["task_type"] == "regression":
        noise = torch.tensor(rng("noise").normal(size=controls["rows"])[list(indices)])
        add(
            "noise",
            "Replayed observation noise",
            [],
            "noise_ratio * original standard normal draw",
            noise * controls["noise_ratio"],
            {"noise_ratio": controls["noise_ratio"], "stream": "noise"},
        )
        targets = signal[:, 0] + controls["noise_ratio"] * noise
        add(
            "target",
            "Regression target",
            ["signal", "noise"],
            "signal + observation noise",
            targets,
            {},
        )
        return targets
    logits = controls["separation"] * signal + torch.tensor(
        controls["class_offsets"] or [0.0] * signal.shape[1], dtype=torch.float64
    )
    add(
        "logits",
        "Class separation and offsets",
        ["signal"],
        "separation * signal + class_offsets",
        logits,
        {"separation": controls["separation"], "class_offsets": controls["class_offsets"]},
    )
    if controls["task_type"] == "binary":
        positive = torch.sigmoid(logits[:, 0])
        probabilities = torch.stack((1 - positive, positive), dim=1)
        formula = "[1 - sigmoid(logit), sigmoid(logit)]"
    else:
        probabilities = torch.softmax(logits, dim=1)
        formula = "softmax(logits)"
    add("probabilities", "Class probabilities", ["logits"], formula, probabilities, {})
    uniforms = torch.tensor(rng("labels").uniform(size=controls["rows"])[list(indices)])
    add(
        "labels",
        "Replayed label draws",
        [],
        "original uniform(0,1) draws",
        uniforms,
        {"stream": "labels"},
    )
    targets = torch.sum(uniforms[:, None] >= probabilities.cumsum(dim=1)[:, :-1], dim=1)
    add(
        "target",
        "Sampled class",
        ["probabilities", "labels"],
        "number of cumulative class probabilities <= uniform draw",
        targets,
        {},
    )
    return targets
