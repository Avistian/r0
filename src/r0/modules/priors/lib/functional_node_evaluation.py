"""One admitted evaluator for frozen DAG generation and original-row inspection."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.exogenous import sample_exogenous
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.frozen_kumo_product import FrozenKumoProductFunction
from r0.modules.priors.lib.frozen_parent_function import FrozenParentFunction
from r0.modules.priors.lib.functional_causal_node import FunctionalCausalNode
from r0.modules.priors.lib.joint_root_base import draw_root_base
from r0.modules.priors.lib.kumo_product_inspection import kumo_product_steps
from r0.modules.priors.lib.node_pipeline import PipelineEvaluation
from r0.modules.priors.lib.parent_function_inspection import parent_function_steps
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger
from r0.modules.priors.lib.signal_noise import signal_noise_scale


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


def evaluate_functional_nodes(
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
    execution: InterventionExecution | None = None,
) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    for index, node in enumerate(nodes):
        bypass = execution.bypass(index, node, rows, device) if execution else None
        if bypass is not None:
            state, readout = bypass
            values.append(state)
            if readouts is not None:
                readouts.append(readout)
            continue
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
            if execution:
                execution.record(index, root=inputs)
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
        if execution:
            execution.record(index, inputs=inputs, function=output, exogenous=raw, scale=scale)
            state, readout = execution.apply(index, node, state, readout)
        if not torch.isfinite(state).all() or not torch.isfinite(readout).all():
            raise ValueError("nonfinite functional causal value; task was not retried")
        _constant_diagnostic(state, index, "state", diagnostics)
        values.append(state)
        if readouts is not None:
            readouts.append(readout)
        if add is not None:
            equation = "f(concat(parent states))" if node.parents else "f(root base)"
            if isinstance(node.function, FrozenParentFunction):
                parents = parent_function_steps(
                    node.function, inputs, node=index, parents=parents, indices=indices, add=add
                )
                equation = f"{node.function.aggregation}(ordered parent members)"
            elif isinstance(node.function, FrozenKumoProductFunction):
                parents = kumo_product_steps(
                    node.function,
                    inputs,
                    prefix=f"kumo-product-{index}",
                    parents=parents,
                    indices=indices,
                    add=add,
                )
                equation = "f_left(same input) * f_right(same input)"
            add(
                f"function-{index}",
                f"Frozen function {index}",
                parents,
                equation,
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
