"""Explicit acyclic structural equations behind the shared task interface."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.calibration import stable_calibration
from r0.modules.priors.lib.causal_evaluation import causal_values as causal_values
from r0.modules.priors.lib.causal_graph import CausalGraph
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.causal_moments import causal_moment_orders
from r0.modules.priors.lib.causal_nodes import ActivatedCausalNode as ActivatedCausalNode
from r0.modules.priors.lib.causal_nodes import CausalNode as CausalNode
from r0.modules.priors.lib.causal_nodes import MatrixCausalNode as MatrixCausalNode
from r0.modules.priors.lib.causal_nodes import StochasticCausalNode as StochasticCausalNode
from r0.modules.priors.lib.causal_nodes import causal_node_from_dict as causal_node_from_dict
from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.node_pipeline import CausalNodePipeline
from r0.modules.priors.lib.pipeline_ancestry import pipeline_ancestry
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


def causal_rng(seed: int, task_id: int, *keys: object) -> np.random.Generator:
    """One versioned stream identity for generation and original-row inspection."""
    return np.random.default_rng(int(record_fingerprint(("causal-v1", seed, task_id, keys)), 16))


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
    def node_pipelines(self) -> tuple[CausalNodePipeline, ...] | None:
        return None

    @property
    def pipeline_working_channels(self) -> int:
        return sum(p.working_extra_channels for p in self.node_pipelines or ())

    @property
    def admission_calibration_rows(self) -> int:
        return self.calibration_rows if self.normalize else 0

    @property
    def sampling_record(self) -> dict[str, Any] | None:
        return None

    @property
    def sampling_parameter_count(self) -> int:
        return 0

    def __post_init__(self) -> None:
        if self.version not in ("causal-v1", "causal-v2") or type(self.task) is not LinearRecipe:
            raise ValueError("causal recipes require shared linear-v1 task controls")
        if type(self.nodes) is not tuple or not self.nodes:
            raise ValueError("causal recipe requires a nonempty node tuple")
        for i, node in enumerate(self.nodes):
            if type(node) not in (
                CausalNode,
                ActivatedCausalNode,
                StochasticCausalNode,
                MatrixCausalNode,
            ) or any(p >= i for p in node.parents):
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
            "activation-exogenous-v1",
            "activation-factorized-v1",
            "activation-importance-v1",
            "activation-matrices-v1",
            "activation-gaussian-matrices-v1",
            "activation-shared-matrices-v1",
            "source-activations-v1",
            "source-shared-matrices-v1",
        ) or (self.construction != "authored" and self.topology is None):
            raise ValueError("sampled construction requires its saved topology")
        if (
            self.construction
            in (
                "gaussian-activation-prior-v1",
                "activation-exogenous-v1",
                "activation-factorized-v1",
                "activation-importance-v1",
                "activation-matrices-v1",
                "activation-gaussian-matrices-v1",
                "activation-shared-matrices-v1",
                "source-activations-v1",
                "source-shared-matrices-v1",
            )
            and type(self) is CausalRecipe
        ):
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
            calibration_rows=self.admission_calibration_rows,
            features=self.task.features,
            outputs=outputs,
            extra_parameters=self.sampling_parameter_count,
            extra_working_channels=self.pipeline_working_channels,
        )
        for node in self.nodes:
            if (
                isinstance(node, MatrixCausalNode)
                and node.weights != node.weight_construction.materialize()
            ):
                raise ValueError("recorded weights disagree with their frozen construction")

        if self.normalize and (
            self.node_pipelines is not None
            or any(isinstance(node, StochasticCausalNode) for node in self.nodes)
        ):
            orders: list[list[float]] = []
            causal_moment_orders(self.nodes, pipelines=self.node_pipelines, readout_orders=orders)
            if any(orders[i][j] <= 2 for i, j in self.target):
                raise ValueError(
                    "finite target second moments are not certified; use normalize=False"
                )


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
    readouts: list[torch.Tensor] = []
    causal_values(
        recipe.nodes,
        recipe.task.rows,
        rng,
        device,
        diagnostics=diagnostics if recipe.version == "causal-v2" else None,
        pipelines=recipe.node_pipelines,
        readout_values=readouts,
    )
    inputs = torch.stack([readouts[i][:, j] for i, j in recipe.observed], dim=1)
    scores = torch.stack([readouts[i][:, j] for i, j in recipe.target], dim=1)
    center, scale = np.zeros(len(recipe.target)), np.ones(len(recipe.target))
    constant_channels: list[int] = []
    if recipe.normalize:
        calibration_values: list[torch.Tensor] = []
        causal_values(
            recipe.nodes,
            recipe.calibration_rows,
            lambda *keys: rng("calibration", *keys),
            "cpu",
            pipelines=recipe.node_pipelines,
            readout_values=calibration_values,
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
        sources: list[set[tuple[int, int]]] = [{(i, j)} for j in range(len(node.bias))]
        if isinstance(node, StochasticCausalNode):
            sources = [
                {(i, source) for source in indices}
                if node.has_noise_scale(j)
                else set[tuple[int, int]]()
                for j, indices in enumerate(node.exogenous.channel_sources(len(node.bias)))
            ]
        projected: list[set[tuple[int, int]]] = []
        for channel in range(len(node.bias)):
            related = set(sources[channel])
            for incoming, parents in enumerate(incoming_ancestry):
                if node.weights[incoming][channel] != 0:
                    related.update(parents)
            projected.append(related)
        pre_output = [set(related) for related in projected]
        if isinstance(node, ActivatedCausalNode):
            if not node.output_activation.uses_input(len(node.bias)):
                projected = (
                    [set() for _ in node.bias]
                    if isinstance(node, StochasticCausalNode)
                    and node.noise_position == "before_output"
                    else [set(source) for source in sources]
                )
            elif node.output_activation.mixes_channels:
                shared = set[tuple[int, int]]().union(*projected)
                if node.noise_position == "after_output":
                    shared = {ref for ref in shared if ref[0] != i}
                projected = [shared | source for source in sources]
        if (
            isinstance(node, StochasticCausalNode)
            and node.noise_position == "after_output"
            and node.noise_response.strength
        ):
            for channel, related in enumerate(projected):
                if sources[channel]:
                    related.update(pre_output[channel])
        for channel, related in enumerate(projected):
            ancestry[i, channel] = related
    if recipe.node_pipelines is not None:
        ancestry = pipeline_ancestry(recipe.nodes, recipe.node_pipelines)
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
            calibration_rows=recipe.admission_calibration_rows,
            features=recipe.task.features,
            outputs=len(recipe.target),
            extra_parameters=recipe.sampling_parameter_count,
            extra_working_channels=recipe.pipeline_working_channels,
        ),
    }
    if any(isinstance(node, StochasticCausalNode) for node in recipe.nodes):
        card["innovation"] = "recorded exogenous laws; independently keyed streams per node/channel"
    if recipe.node_pipelines is not None or any(
        isinstance(node, StochasticCausalNode) for node in recipe.nodes
    ):
        orders: list[list[float]] = []
        causal_moment_orders(recipe.nodes, pipelines=recipe.node_pipelines, readout_orders=orders)
        card["moment_contract"] = {
            "target_orders": [
                "all" if orders[i][j] == float("inf") else orders[i][j] for i, j in recipe.target
            ],
            "meaning": (
                "positive absolute moments strictly below each bound are certified finite; "
                "zero means unknown"
            ),
            "normalization_requires": "certified finite second moments",
        }
    if recipe.version == "causal-v2":
        card["activation_diagnostics"] = diagnostics
    if recipe.node_pipelines is not None:
        card["pipelines"] = [asdict(p) for p in recipe.node_pipelines]
        card["active_definition"] = (
            "potential exogenous dependence of explicit readouts; excludes later state noise"
        )
        card["readout_contract"] = (
            "explicit converter channels before innovation/node importance; "
            "other channels use final state"
        )
    if recipe.sampling_record is not None:
        card["activation_sampling"] = recipe.sampling_record
    return complete_task(
        recipe.task, asdict(recipe), card, inputs, scores, seed=seed, task_id=task_id, rng=rng
    )
