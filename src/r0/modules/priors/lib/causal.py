"""Explicit acyclic structural equations behind the shared task interface."""

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.calibration import stable_calibration
from r0.modules.priors.lib.causal_activations import (
    CausalActivation,
    activate,
    activation_diagnostic,
)
from r0.modules.priors.lib.causal_graph import CausalGraph
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


def causal_rng(seed: int, task_id: int, *keys: object) -> np.random.Generator:
    """One versioned stream identity for generation and original-row inspection."""
    return np.random.default_rng(int(record_fingerprint(("causal-v1", seed, task_id, keys)), 16))


@dataclass(frozen=True, slots=True)
class CausalNode:
    parents: tuple[int, ...] = ()
    weights: tuple[tuple[float, ...], ...] = ()
    bias: tuple[float, ...] = (0.0,)
    noise_scale: tuple[float, ...] = (1.0,)

    @property
    def activation_parameter_count(self) -> int:
        return 0

    def __post_init__(self) -> None:
        if (
            type(self.parents) is not tuple
            or any(type(p) is not int or p < 0 for p in self.parents)
            or len(set(self.parents)) != len(self.parents)
        ):
            raise ValueError("causal parents must be unique nonnegative node indices")
        if (
            type(self.bias) is not tuple
            or not self.bias
            or type(self.noise_scale) is not tuple
            or len(self.noise_scale) != len(self.bias)
            or type(self.weights) is not tuple
            or any(type(row) is not tuple or len(row) != len(self.bias) for row in self.weights)
        ):
            raise ValueError("causal bias, noise and weight output widths must agree")
        values = (*self.bias, *self.noise_scale, *(v for row in self.weights for v in row))
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError("causal parameters must be finite numbers")
        if any(v < 0 for v in self.noise_scale):
            raise ValueError("structural noise scales must be nonnegative")


@dataclass(frozen=True, slots=True)
class ActivatedCausalNode(CausalNode):
    input_activation: CausalActivation = CausalActivation()
    output_activation: CausalActivation = CausalActivation()
    noise_position: str = "before_output"

    @property
    def activation_parameter_count(self) -> int:
        return len(self.input_activation.parameters) + len(self.output_activation.parameters)

    def __post_init__(self) -> None:
        CausalNode.__post_init__(self)
        if any(
            type(a) is not CausalActivation
            for a in (
                self.input_activation,
                self.output_activation,
            )
        ):
            raise ValueError("node activations must use CausalActivation")
        if self.noise_position not in ("before_output", "after_output"):
            raise ValueError("noise position must be before_output or after_output")
        if not self.parents and self.input_activation != CausalActivation():
            raise ValueError("root nodes have no parent channels to activate")


def causal_node_from_dict(data: dict[str, Any]) -> CausalNode:
    """One decoder shared by recipe loading and original-row inspection."""
    fields = dict(data)
    for key in ("parents", "bias", "noise_scale"):
        if key in fields:
            fields[key] = tuple(fields[key])
    if "weights" in fields:
        fields["weights"] = tuple(tuple(row) for row in fields["weights"])
    activated = any(
        key in fields
        for key in (
            "input_activation",
            "output_activation",
            "noise_position",
        )
    )
    for key in ("input_activation", "output_activation"):
        if key in fields:
            activation = dict(fields[key])
            activation["parameters"] = tuple(
                tuple(pair) for pair in activation.get("parameters", ())
            )
            fields[key] = CausalActivation(**activation)
    return ActivatedCausalNode(**fields) if activated else CausalNode(**fields)


@dataclass(frozen=True, slots=True)
class CausalRecipe:
    task: LinearRecipe
    nodes: tuple[CausalNode, ...]
    observed: tuple[tuple[int, int], ...]
    target: tuple[tuple[int, int], ...]
    normalize: bool = False
    calibration_rows: int = 256
    limits: CausalLimits = CausalLimits()
    topology: CausalGraph | None = None
    construction: str = "authored"
    version: str = "causal-v1"

    @property
    def sampling_record(self) -> dict[str, Any] | None:
        return None

    def __post_init__(self) -> None:
        if self.version not in ("causal-v1", "causal-v2") or type(self.task) is not LinearRecipe:
            raise ValueError("causal recipes require shared linear-v1 task controls")
        if type(self.nodes) is not tuple or not self.nodes:
            raise ValueError("causal recipe requires a nonempty node tuple")
        for i, node in enumerate(self.nodes):
            if type(node) not in (CausalNode, ActivatedCausalNode) or any(
                p >= i for p in node.parents
            ):
                raise ValueError("causal nodes must be in acyclic parent-before-child order")
            if isinstance(node, ActivatedCausalNode) and self.version != "causal-v2":
                raise ValueError("activated nodes require causal-v2")
            incoming = sum(len(self.nodes[p].bias) for p in node.parents)
            if len(node.weights) != incoming:
                raise ValueError("causal weight input width must match all parent channels")
        if self.topology is not None and (
            type(self.topology) is not CausalGraph
            or self.topology.parents != tuple(node.parents for node in self.nodes)
        ):
            raise ValueError("saved topology must match structural node parents")
        if self.construction not in (
            "authored",
            "gaussian-affine-v1",
            "gaussian-activated-v1",
            "gaussian-activation-prior-v1",
        ) or (self.construction != "authored" and self.topology is None):
            raise ValueError("sampled construction requires its saved topology")
        if self.construction == "gaussian-activation-prior-v1" and type(self) is CausalRecipe:
            raise ValueError("activation sampling construction requires its saved prior")
        for refs in (self.observed, self.target):
            if type(refs) is not tuple or any(
                type(ref) is not tuple
                or len(ref) != 2
                or any(type(v) is not int for v in ref)
                or not 0 <= ref[0] < len(self.nodes)
                or not 0 <= ref[1] < len(self.nodes[ref[0]].bias)
                for ref in refs
            ):
                raise ValueError("readouts must identify valid node/channel pairs")
        outputs = self.task.classes if self.task.task_type == "multiclass" else 1
        if len(self.observed) != self.task.features or len(self.target) != outputs:
            raise ValueError("causal readout widths must match feature and target controls")
        if set(self.observed) & set(self.target):
            raise ValueError("a declared target channel cannot be observed as a feature")
        if type(self.normalize) is not bool or (
            type(self.calibration_rows) is not int or not 32 <= self.calibration_rows <= 65536
        ):
            raise ValueError("normalization requires a boolean and 32..65536 calibration rows")
        if type(self.limits) is not CausalLimits:
            raise ValueError("causal limits must use CausalLimits")
        self.limits.admit(
            self.nodes,
            rows=self.task.rows,
            calibration_rows=self.calibration_rows if self.normalize else 0,
            features=self.task.features,
            outputs=outputs,
        )


def causal_values(
    nodes: tuple[CausalNode, ...],
    rows: int,
    rng: Callable[..., np.random.Generator],
    device: str,
    indices: Sequence[int] | None = None,
    *,
    diagnostics: list[dict[str, Any]] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Replay selected original rows without retaining all intermediate task values."""
    values: list[torch.Tensor] = []
    innovations: list[torch.Tensor] = []
    for i, node in enumerate(nodes):
        noise = np.column_stack(
            [rng("innovation", i, j).normal(size=rows) for j in range(len(node.bias))]
        )
        if indices is not None:
            noise = noise[list(indices)]
        innovation = torch.tensor(noise, dtype=torch.float64, device=device) * torch.tensor(
            node.noise_scale, dtype=torch.float64, device=device
        )
        value = innovation + torch.tensor(node.bias, dtype=torch.float64, device=device)
        if isinstance(node, ActivatedCausalNode) and node.noise_position == "after_output":
            value = torch.zeros_like(innovation) + torch.tensor(
                node.bias, dtype=torch.float64, device=device
            )
        if node.parents:
            parents = torch.cat([values[p] for p in node.parents], dim=1)
            if isinstance(node, ActivatedCausalNode):
                if diagnostics is not None:
                    diagnostic = activation_diagnostic(parents, node.input_activation)
                    if diagnostic is not None:
                        diagnostics.append({"node": i, "stage": "input", **diagnostic})
                parents = activate(parents, node.input_activation)
            value = value + parents @ torch.tensor(node.weights, dtype=torch.float64, device=device)
        if isinstance(node, ActivatedCausalNode):
            if diagnostics is not None:
                diagnostic = activation_diagnostic(value, node.output_activation)
                if diagnostic is not None:
                    diagnostics.append({"node": i, "stage": "output", **diagnostic})
            value = activate(value, node.output_activation)
            if node.noise_position == "after_output":
                value = value + innovation
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite causal node {i}; task was not retried")
        values.append(value)
        innovations.append(innovation)
    return values, innovations


def generate_causal(
    recipe: CausalRecipe, *, seed: int, task_id: int, device: str = "cpu"
) -> TrainingTask:
    """Sample structural innovations, evaluate the DAG, then adapt targets."""
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; no implicit fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select CPU explicitly")

    def rng(*keys: object) -> np.random.Generator:
        return causal_rng(seed, task_id, *keys)

    diagnostics: list[dict[str, Any]] = []
    values, _ = causal_values(
        recipe.nodes,
        recipe.task.rows,
        rng,
        device,
        diagnostics=diagnostics if recipe.version == "causal-v2" else None,
    )
    inputs = torch.stack([values[i][:, j] for i, j in recipe.observed], dim=1)
    scores = torch.stack([values[i][:, j] for i, j in recipe.target], dim=1)
    center, scale = np.zeros(len(recipe.target)), np.ones(len(recipe.target))
    constant_channels: list[int] = []
    if recipe.normalize:
        calibration_values, _ = causal_values(
            recipe.nodes,
            recipe.calibration_rows,
            lambda *keys: rng("calibration", *keys),
            "cpu",
        )
        calibration_raw = torch.stack(
            [calibration_values[i][:, j] for i, j in recipe.target], dim=1
        )
        raw = np.asarray(calibration_raw.numpy(), dtype=np.float64)
        for channel in range(raw.shape[1]):
            column = raw[:, channel : channel + 1]
            if np.all(column == column[0]):
                constant_channels.append(channel)
            else:
                column_center, column_scale = stable_calibration(column)
                center[channel], scale[channel] = column_center[0], column_scale[0]
        scores = (scores - torch.tensor(center, device=device)) / torch.tensor(scale, device=device)
    # Channel ancestry is a structural descriptor, not an empirical learnability claim.
    ancestry: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for i, node in enumerate(recipe.nodes):
        parent_channels = [(p, j) for p in node.parents for j in range(len(recipe.nodes[p].bias))]
        incoming_ancestry: list[set[tuple[int, int]]] = [
            ancestry[parent] for parent in parent_channels
        ]
        if isinstance(node, ActivatedCausalNode):
            if not node.input_activation.uses_input(len(parent_channels)):
                incoming_ancestry = [set[tuple[int, int]]() for _ in parent_channels]
            elif node.input_activation.mixes_channels:
                shared = set[tuple[int, int]]().union(*incoming_ancestry)
                incoming_ancestry = [shared for _ in parent_channels]
        projected: list[set[tuple[int, int]]] = []
        for channel in range(len(node.bias)):
            related = {(i, channel)}
            for incoming, parents in enumerate(incoming_ancestry):
                if node.weights[incoming][channel] != 0:
                    related.update(parents)
            projected.append(related)
        if isinstance(node, ActivatedCausalNode):
            if not node.output_activation.uses_input(len(node.bias)):
                projected = [{(i, channel)} for channel in range(len(node.bias))]
            elif node.output_activation.mixes_channels:
                shared = set[tuple[int, int]]().union(*projected)
                if node.noise_position == "after_output":
                    shared = {ref for ref in shared if ref[0] != i}
                projected = [shared | {(i, channel)} for channel in range(len(node.bias))]
        for channel, related in enumerate(projected):
            ancestry[i, channel] = related
    target_ancestors: set[tuple[int, int]] = set()
    for ref in recipe.target:
        target_ancestors.update(ancestry[ref])
    active = [i for i, ref in enumerate(recipe.observed) if ancestry[ref] & target_ancestors]
    flags: list[str] = []
    if not active:
        flags.append("disconnected_target")
    if torch.all(scores == scores[0]).item():
        flags.append("zero_target_dispersion")
    flags.extend(sorted({flag for diagnostic in diagnostics for flag in diagnostic["flags"]}))
    card = {
        "version": recipe.version,
        "family": "causal",
        "topology": asdict(recipe.topology) if recipe.topology is not None else None,
        "construction": recipe.construction,
        "nodes": [asdict(node) for node in recipe.nodes],
        "observed": recipe.observed,
        "target": recipe.target,
        "active": active,
        "active_definition": (
            "declared projection paths and channel-coupled activations; potential dependence only"
            if recipe.version == "causal-v2"
            else "shared nonzero-edge channel ancestry; potential dependence only"
        ),
        "flags": flags,
        "normalization": "fixed independent calibration" if recipe.normalize else "none",
        "calibration": {
            "rows": recipe.calibration_rows if recipe.normalize else 0,
            "center": center.tolist(),
            "scale": scale.tolist(),
            "constant_channels": constant_channels,
        },
        "innovation": "independent standard Gaussian per node/channel; keyed by seed/task",
        "seed": seed,
        "task_id": task_id,
        "resource_limits": asdict(recipe.limits),
        "resource_admission": recipe.limits.admit(
            recipe.nodes,
            rows=recipe.task.rows,
            calibration_rows=recipe.calibration_rows if recipe.normalize else 0,
            features=recipe.task.features,
            outputs=len(recipe.target),
        ),
    }
    if recipe.version == "causal-v2":
        card["activation_diagnostics"] = diagnostics
    if recipe.sampling_record is not None:
        card["activation_sampling"] = recipe.sampling_record
    return complete_task(
        recipe.task, asdict(recipe), card, inputs, scores, seed=seed, task_id=task_id, rng=rng
    )
