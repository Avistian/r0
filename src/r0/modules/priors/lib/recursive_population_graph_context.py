"""Per-node runtime ownership, path traces and conditional certificates for populations."""

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from r0.modules.priors.lib.frozen_function_evaluation import (
    FrozenFunction,
    evaluate_admitted_function,
)
from r0.modules.priors.lib.functional_analysis import ChannelCertificate, function_certificates
from r0.modules.priors.lib.recursive_forest_evaluation import (
    evaluate_recursive_forest,
    inspect_recursive_forest,
)
from r0.modules.priors.lib.recursive_forest_sampling import SampledRecursiveForest


class RecursivePopulationContext:
    """Only the explicit enclosing graph admits/constructs this ephemeral evaluation context."""

    def __init__(
        self, samples: dict[int, SampledRecursiveForest], *, disabled: frozenset[int] = frozenset()
    ) -> None:
        self.samples = samples
        self.disabled = disabled
        self.inputs: dict[int, torch.Tensor] = {}
        self.outputs: dict[int, torch.Tensor] = {}
        self.stages: dict[int, dict[str, torch.Tensor]] = {}

    def record(self, index: int, **values: torch.Tensor) -> None:
        self.stages.setdefault(index, {}).update({k: v.detach().clone() for k, v in values.items()})

    def replaces(self, index: int) -> bool:
        return index in self.samples and index not in self.disabled

    def evaluate(
        self,
        index: int,
        function: FrozenFunction,
        inputs: torch.Tensor,
        diagnostics: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        self.inputs[index] = inputs.detach().clone()
        if not self.replaces(index):
            output = evaluate_admitted_function(function, inputs, diagnostics=diagnostics)
        else:
            sample = self.samples[index]
            raw = evaluate_recursive_forest(
                sample, inputs.detach().cpu().numpy(), limits=sample.limits
            )
            if sample.plan.classes:
                labels = torch.tensor(raw, dtype=torch.int64, device=inputs.device)
                output = torch.nn.functional.one_hot(labels, num_classes=sample.plan.classes).to(
                    torch.float64
                )
            else:
                output = torch.tensor(raw, dtype=torch.float64, device=inputs.device)
        self.outputs[index] = output.detach().clone()
        return output

    def metadata(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "population_plan": asdict(sample.plan),
            "original_rows": sample.plan.populations,
            "construction": "complete per-node sampled forest receipt in enclosing archive",
            "output_semantics": "actual majority-label indicator"
            if sample.plan.classes
            else "coordinatewise regression mean",
            "lifecycle": "fresh-forest-per-row",
        }

    def certify(
        self, index: int, function: FrozenFunction, incoming: list[ChannelCertificate]
    ) -> list[ChannelCertificate]:
        if not self.replaces(index):
            return function_certificates(function, incoming)
        sample = self.samples[index]
        used = {
            node.feature
            for forest in sample.forests
            for tree in forest.trees
            for node in tree.nodes
            if node.left >= 0
        }
        sources = frozenset[tuple[int, int]]().union(*(incoming[j].sources for j in used))
        # Conditional on all retained finite leaves. Construction randomness is separately retained.
        return [ChannelCertificate(sources)] * function.output_width

    def steps(
        self, index: int, *, parents: list[str], indices: Sequence[int], add: Callable[..., None]
    ) -> list[str]:
        sample = self.samples[index]
        inputs = self.inputs[index]
        paths = inspect_recursive_forest(
            sample, inputs.detach().cpu().numpy(), limits=sample.limits
        )
        ids: list[str] = []
        for tree in range(sample.plan.estimators):
            leaves = [
                sample.forests[row].trees[tree].nodes[paths[row][tree][-1]] for row in indices
            ]
            values = (
                [[n.label] for n in leaves] if sample.plan.classes else [n.value for n in leaves]
            )
            identifier = f"population-{index}-tree-{tree}"
            add(
                identifier,
                f"Node {index} row-owned tree {tree}",
                parents,
                "each original row follows its own retained tree; equality goes left",
                torch.tensor(values, dtype=torch.float64),
                {
                    "original_rows": list(indices),
                    "paths": [paths[row][tree] for row in indices],
                    "leaf_semantics": "actual class label"
                    if sample.plan.classes
                    else "regression vector",
                },
            )
            ids.append(identifier)
        identifier = f"population-{index}-aggregate"
        add(
            identifier,
            f"Node {index} row-owned forest aggregation",
            ids,
            "majority vote to exact winner indicator"
            if sample.plan.classes
            else "coordinatewise mean",
            self.outputs[index][list(indices)],
            self.metadata(index),
        )
        return [identifier]
