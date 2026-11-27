"""Architecture arithmetic and optional per-stage trace share one frozen evaluator."""

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.causal_activations import activation_diagnostic
from r0.modules.priors.lib.structural_network import FrozenNetworkFunction, NetworkAffine


def network_geometry(function: FrozenNetworkFunction) -> tuple[tuple[int, ...], int, int]:
    widths = (function.input_width, *(b.output_width for b in function.blocks))
    work = 8 * sum(widths)
    for block in function.blocks:
        work += 8 * block.experts[0].input_width + 12 * block.output_width * len(block.experts)
        work += 8 * len(block.experts) + 8 * len(block.skips) * block.output_width
    dimensions = list(widths)
    for block in function.blocks:
        dimensions.append(block.experts[0].input_width)
        if block.gate:
            dimensions.append(len(block.experts))
    return tuple(dimensions), work, 0


def evaluate_network(
    function: FrozenNetworkFunction,
    inputs: torch.Tensor,
    *,
    trace: Callable[[str, torch.Tensor, dict[str, Any]], None] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    def retain(name: str, value: torch.Tensor, record: object) -> torch.Tensor:
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite structural network {name}; evaluation was not retried")
        if trace:
            trace(name, value, {"record": record})
        return value

    def activate(branch: NetworkAffine, value: torch.Tensor, name: str) -> torch.Tensor:
        if diagnostics is not None:
            entry = activation_diagnostic(value, branch.activation)
            if entry is not None:
                diagnostics.append({"stage": name, **entry})
        return branch.activation.apply(value)

    def affine(branch: NetworkAffine, value: torch.Tensor, name: str) -> torch.Tensor:
        if branch.placement == "before":
            value = retain(
                name + "-activation", activate(branch, value, name), asdict(branch.activation)
            )
        matrix = value.new_tensor(branch.matrix)
        if branch.mask:
            matrix = torch.where(torch.tensor(branch.mask, device=value.device), matrix, 0.0)
        matrix = matrix * branch.multiplier
        if not torch.isfinite(matrix).all():
            raise ValueError("nonfinite structural network weight scaling")
        result = retain(
            name + "-affine", value @ matrix.T + value.new_tensor(branch.bias), asdict(branch)
        )
        if branch.placement == "after":
            result = retain(
                name + "-activation", activate(branch, result, name), asdict(branch.activation)
            )
        return result

    values = [inputs]
    for index, block in enumerate(function.blocks, 1):
        name = f"block-{index}"
        argument = retain(
            name + "-inputs",
            torch.cat([values[p] for p in block.parents], dim=1),
            {"parents": block.parents},
        )
        experts = [affine(e, argument, f"{name}-expert-{j}") for j, e in enumerate(block.experts)]
        if block.gate:
            logits = affine(block.gate, argument, name + "-gate")
            weights = retain(
                name + "-weights", logits.softmax(dim=1), {"law": "input-dependent softmax"}
            )
            result = (torch.stack(experts, dim=1) * weights[:, :, None]).sum(dim=1)
        else:
            result = experts[0]
        for j, skip in enumerate(block.skips):
            result = result + affine(skip.affine, values[skip.source], f"{name}-skip-{j}")
        values.append(
            retain(
                name + "-state",
                result,
                {"parents": block.parents, "skip_sources": [s.source for s in block.skips]},
            )
        )
    return values[-1]
