"""Versioned input masks and extra observed columns around an unchanged network."""

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.frozen_function_evaluation import evaluate_admitted_function
from r0.modules.priors.lib.functional_causal_evaluation import (
    evaluate_functional_prefix,
    functional_rng,
)
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.network_recipe import NetworkRecipe
from r0.modules.priors.lib.network_sampling_records import SampledNetwork
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


@dataclass(frozen=True, slots=True)
class NetworkInputView:
    base: NetworkRecipe
    task: LinearRecipe
    active_inputs: tuple[int, ...]
    version: str = "network-input-view-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "network-input-view-v1"
            or type(self.base) is not NetworkRecipe
            or type(self.task) is not LinearRecipe
            or type(self.active_inputs) is not tuple
            or not self.active_inputs
            or any(type(i) is not int for i in self.active_inputs)
        ):
            raise ValueError("network input view requires a frozen base and explicit active inputs")
        width = self.base.function.input_width
        if (
            self.active_inputs != tuple(sorted(set(self.active_inputs)))
            or any(not 0 <= i < width for i in self.active_inputs)
            or self.task.features < width
            or self.task.signal_features != width
            or self.task.active_features != len(self.active_inputs)
            or self.task.task_type != self.base.task.task_type
            or self.task.classes != self.base.task.classes
        ):
            raise ValueError("network view geometry, active counts or target domain disagree")
        self.admit()

    @property
    def limits(self) -> CausalLimits:
        return self.base.limits

    @property
    def runtime_base(self) -> NetworkRecipe:
        width = self.base.function.input_width
        return replace(
            self.base,
            task=replace(self.task, features=width, signal_features=width, active_features=width),
        )

    def admit(self, *, rows: int | None = None) -> dict[str, int]:
        count = self.task.rows if rows is None else rows
        if type(count) is not int or count < 1:
            raise ValueError("view evaluation rows must be positive integers")
        report = self.runtime_base.runtime.admit(evaluation_rows=count)
        # Original generation and masked evaluation retain several input/output
        # arrays concurrently; metadata of the immutable base is already charged.
        extra = (
            8
            * max(count, self.task.rows)
            * (4 * self.task.features + 4 * self.base.function.output_width)
        )
        if report["estimated_working_bytes"] + extra > self.limits.working_bytes:
            raise ValueError("network view aggregate working limit exceeded")
        return {**report, "estimated_working_bytes": report["estimated_working_bytes"] + extra}


def view_signal(
    view: NetworkInputView,
    inputs: torch.Tensor,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    view.admit(rows=len(inputs))
    if (
        inputs.ndim != 2
        or inputs.shape[1] != view.task.features
        or inputs.dtype != torch.float64
        or not torch.isfinite(inputs).all()
    ):
        raise ValueError("network view requires finite aligned float64 observed inputs")
    width = view.base.function.input_width
    mask = torch.zeros(width, dtype=torch.bool, device=inputs.device)
    mask[list(view.active_inputs)] = True
    argument = torch.where(mask, inputs[:, :width], 0.0)
    return evaluate_admitted_function(view.base.function, argument, diagnostics=diagnostics)


def generate_network_view(
    view: NetworkInputView, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    view.admit()
    if isinstance(view.base.network, SampledNetwork):
        view.base.network.validate_replay()
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    roots = evaluate_functional_prefix(
        FunctionalPrefix(view.runtime_base.runtime.nodes[:1], view.task.rows, view.limits),
        rng=lambda *keys: functional_rng(seed, task_id, *keys),
        device=device,
    )
    width = view.base.function.input_width
    columns = [roots.readouts[0]]
    for index in range(width, view.task.features):
        values = functional_rng(seed, task_id, "network-extra-column-v1", index).standard_normal(
            view.task.rows
        )
        columns.append(torch.tensor(values[:, None], dtype=torch.float64, device=device))
    inputs = torch.cat(columns, dim=1)
    diagnostics: list[dict[str, Any]] = []
    scores = view_signal(view, inputs, diagnostics=diagnostics)
    flags = sorted({flag for entry in diagnostics for flag in entry["flags"]})
    if bool(torch.all(scores == scores[0])):
        flags.append("zero_target_dispersion")
    card = dict(
        family="standalone_network",
        architecture="unchanged network block DAG behind an explicit input view",
        flags=flags,
        node_diagnostics=diagnostics,
        version=view.version,
        active=list(view.active_inputs),
        active_definition="explicit network input mask; potential dependence only",
        input_view={
            "network_columns": list(range(width)),
            "active_inputs": list(view.active_inputs),
            "fill": 0.0,
            "position": "after original joint root transformation",
            "extra_columns": "independently keyed standard Gaussian columns",
        },
        normalization="none; input masking does not preserve signal variance",
        resource_admission=view.admit(),
    )
    return complete_task(
        view.task,
        asdict(view),
        card,
        inputs,
        scores,
        seed=seed,
        task_id=task_id,
        rng=lambda *keys: functional_rng(seed, task_id, *keys),
    )


def network_view_from_dict(data: dict[str, Any], *, limits: CausalLimits) -> NetworkInputView:
    from r0.modules.priors.lib.generate import recipe_from_dict
    from r0.modules.priors.lib.network_recipe import network_recipe_from_dict

    if set(data) != {"base", "task", "active_inputs", "version"}:
        raise ValueError("network input view fields differ")
    try:
        base = network_recipe_from_dict(data["base"], limits=limits)
        task = recipe_from_dict(data["task"], limits=limits)
        if type(task) is not LinearRecipe or type(data["active_inputs"]) not in (list, tuple):
            raise ValueError("network input view task or active input record differs")
        return NetworkInputView(base, task, tuple(data["active_inputs"]), data["version"])
    except (TypeError, KeyError, AttributeError, OverflowError) as error:
        raise ValueError("malformed network input view") from error
