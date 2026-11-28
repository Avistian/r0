"""Reconstruct a bounded generation graph from saved parameters and random streams."""

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal import causal_rng
from r0.modules.priors.lib.causal_inspection import causal_steps
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.functional_causal_evaluation import functional_rng
from r0.modules.priors.lib.functional_steps import functional_steps
from r0.modules.priors.lib.graph_functions import analytic_steps, composite_steps, gp_steps
from r0.modules.priors.lib.inspection_preview import preview_parameters as _preview
from r0.modules.priors.lib.native_inspection import native_steps
from r0.modules.priors.lib.native_vector_inspection import native_vector_steps
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.signals import component_inputs, multiply_signals
from r0.modules.priors.lib.target_inspection import target_steps as target_steps
from r0.modules.priors.lib.training_task import TrainingTask


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
    record = json.loads(task.recipe_json)
    raw = record.get("recipe", {})
    if raw.get("version") == "population-xgboost-graph-v1":
        from r0.modules.priors.lib.xgboost_population_inspection import explain_xgboost_population

        return explain_xgboost_population(
            task, indices, component=component, column_start=column_start, limits=limits
        )

    if raw.get("version") == "network-observation-recipe-v1" or (
        raw.get("version") == "causal-intervention-v1"
        and raw.get("base", {}).get("version") == "network-observation-recipe-v1"
    ):
        from r0.modules.priors.lib.observation_task_validation import validate_observation_task

        validate_observation_task(task, record, limits=limits or CausalLimits())
    if raw.get("version") in ("sampled-hyperprior-graph-v1", "sampled-joint-hyperprior-graph-v1"):
        from r0.modules.priors.lib.hyperprior_task_validation import validate_hyperprior_task

        validate_hyperprior_task(task, record, limits=limits or CausalLimits())
    if not 1 <= len(indices) <= 16:
        raise ValueError("select 1..16 original rows")
    if any(type(i) is not int or not 0 <= i < len(task.inputs) for i in indices):
        raise ValueError("invalid original row index")
    if type(column_start) is not int or not 0 <= column_start <= 4095:
        raise ValueError("column_start must be 0..4095")
    if raw.get("version") in (
        "sampled-typed-topology-r0-v1",
        "sampled-typed-parent-topology-r0-v1",
    ):
        from r0.modules.priors.lib.typed_topology_output import typed_topology_trace

        if component is not None:
            raise ValueError("typed topology has no composite children")
        return typed_topology_trace(
            task, indices, limits=limits or CausalLimits(), column_start=column_start
        )
    card = json.loads(task.mechanism_json)
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

    from r0.modules.priors.lib.gp_inspection_dispatch import gp_inspection

    gp_trace = gp_inspection(task, raw, indices, add, component=component, limits=limits)
    if gp_trace is not None:
        return {**gp_trace, "nodes": nodes}

    from r0.modules.priors.lib.population_task_semantics import is_population_task_record

    if is_population_task_record(record, card):
        from r0.modules.priors.lib.population_inspection_dispatch import population_inspection_steps

        if component is not None:
            raise ValueError("recursive populations do not have composite children")
        return {**population_inspection_steps(task, indices, add, limits=limits), "nodes": nodes}
    inputs = task.inputs.detach().cpu()[list(indices)].clone()
    if record["recipe"].get("version") in ("converted-causal-v1", "sampled-converted-graph-v1"):
        from r0.modules.priors.lib.converted_inspection import converted_steps

        if component is not None:
            raise ValueError("converted graphs do not have composite children")
        return {**converted_steps(task, indices, add, limits=limits), "nodes": nodes}
    if record["recipe"].get("version") in ("network-input-view-v1", "network-input-view-v2"):
        from r0.modules.priors.lib.network_view_inspection import network_view_steps

        if component is not None:
            raise ValueError("network input views do not have composite children")
        return {**network_view_steps(task, indices, add, limits=limits), "nodes": nodes}

    if record["recipe"].get("version") == "prototype-batch-network-realized-v1":
        from r0.modules.priors.lib.prototype_batch_inspection import prototype_batch_steps

        if component is not None:
            raise ValueError("prototype batch networks do not have composite children")
        return {**prototype_batch_steps(task, indices, add, limits=limits), "nodes": nodes}

    if record["recipe"].get("version") == "standalone-network-v1":
        from r0.modules.priors.lib.network_inspection import standalone_network_steps

        if component is not None:
            raise ValueError("standalone networks do not have composite children")
        return {**standalone_network_steps(task, indices, add, limits=limits), "nodes": nodes}

    if record["recipe"].get("version") == "sampled-typed-intervention-policy-v1":
        from r0.modules.priors.lib.typed_policy_output import typed_policy_steps

        if component is not None:
            raise ValueError("typed policies do not have composite children")
        return {**typed_policy_steps(task, indices, add, limits=limits), "nodes": nodes}
    if record["recipe"].get("version") == "sampled-intervention-policy-v1":
        from r0.modules.priors.lib.intervention_policy_output import policy_steps

        if component is not None:
            raise ValueError("sampled policies do not have composite children")
        return {**policy_steps(task, indices, add, limits=limits), "nodes": nodes}

    if record["recipe"].get("version") == "causal-intervention-v1":
        from r0.modules.priors.lib.causal_intervention_inspection import intervention_steps

        if component is not None:
            raise ValueError("interventions do not have composite children")
        return {**intervention_steps(task, indices, add, limits=limits), "nodes": nodes}
    if record["recipe"].get("version") in (
        "native-vector-causal-v1",
        "sampled-native-vector-graph-v1",
    ):
        if component is not None:
            raise ValueError("native vector graphs do not have composite children")
        return {**native_vector_steps(task, indices, add, limits=limits), "nodes": nodes}

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
            "standalone-network-innovations-v1",
            "network-observation-recipe-v1",
            "sampled-hyperprior-graph-v1",
            "sampled-joint-hyperprior-graph-v1",
            "fitted-functional-causal-v1",
            "xgboost-fitted-functional-causal-v1",
            "sampled-xgboost-functional-causal-v1",
            "sampled-fitted-functional-causal-v1",
            "population-fitted-functional-causal-v1",
            "sampled-population-fitted-functional-causal-v1",
            "support-population-functional-causal-v1",
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
            "standalone-network-innovations-v1",
            "network-observation-recipe-v1",
            "sampled-hyperprior-graph-v1",
            "sampled-joint-hyperprior-graph-v1",
            "fitted-functional-causal-v1",
            "xgboost-fitted-functional-causal-v1",
            "sampled-xgboost-functional-causal-v1",
            "sampled-fitted-functional-causal-v1",
            "population-fitted-functional-causal-v1",
            "sampled-population-fitted-functional-causal-v1",
            "support-population-functional-causal-v1",
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
                    "standalone-network-innovations-v1",
                    "network-observation-recipe-v1",
                    "sampled-hyperprior-graph-v1",
                    "sampled-joint-hyperprior-graph-v1",
                    "fitted-functional-causal-v1",
                    "xgboost-fitted-functional-causal-v1",
                    "sampled-xgboost-functional-causal-v1",
                    "sampled-fitted-functional-causal-v1",
                    "population-fitted-functional-causal-v1",
                    "sampled-population-fitted-functional-causal-v1",
                    "support-population-functional-causal-v1",
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
        "standalone-network-innovations-v1",
        "network-observation-recipe-v1",
        "sampled-hyperprior-graph-v1",
        "sampled-joint-hyperprior-graph-v1",
        "fitted-functional-causal-v1",
        "xgboost-fitted-functional-causal-v1",
        "sampled-xgboost-functional-causal-v1",
        "sampled-fitted-functional-causal-v1",
        "population-fitted-functional-causal-v1",
        "sampled-population-fitted-functional-causal-v1",
        "support-population-functional-causal-v1",
        "functional-causal-v2",
        "sampled-functional-graph-v1",
        "sampled-parent-graph-v1",
        "sampled-topology-graph-v1",
        "sampled-geometry-graph-v1",
    )
    observed_step = nodes.pop(0) if causal else None
    if card["version"] in (
        "functional-causal-v1",
        "standalone-network-innovations-v1",
        "network-observation-recipe-v1",
        "sampled-hyperprior-graph-v1",
        "sampled-joint-hyperprior-graph-v1",
        "fitted-functional-causal-v1",
        "xgboost-fitted-functional-causal-v1",
        "sampled-xgboost-functional-causal-v1",
        "sampled-fitted-functional-causal-v1",
        "population-fitted-functional-causal-v1",
        "sampled-population-fitted-functional-causal-v1",
        "support-population-functional-causal-v1",
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
            "sampled-hyperprior-graph-v1",
            "sampled-joint-hyperprior-graph-v1",
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
            "standalone-network-innovations-v1",
            "network-observation-recipe-v1",
            "sampled-hyperprior-graph-v1",
            "sampled-joint-hyperprior-graph-v1",
            "fitted-functional-causal-v1",
            "xgboost-fitted-functional-causal-v1",
            "sampled-xgboost-functional-causal-v1",
            "sampled-fitted-functional-causal-v1",
            "population-fitted-functional-causal-v1",
            "sampled-population-fitted-functional-causal-v1",
            "support-population-functional-causal-v1",
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

    targets = target_steps(controls, signal, indices, rng, add)
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
        **(
            {"observation_selection": card["observation_selection"]}
            if "observation_selection" in card
            else {}
        ),
        "component": None,
        "component_count": len(components),
        "nodes": nodes,
        "signal_max_error": float((signal - expected).abs().max()),
        "target_max_error": float((targets - original_targets).abs().max()),
    }
