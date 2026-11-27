"""One admitted evaluator for frozen DAG generation and original-row inspection."""

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from r0.modules.priors.lib.causal_intervention_execution import InterventionExecution
from r0.modules.priors.lib.construction_provenance import construction_source_config
from r0.modules.priors.lib.fitted_tree_metadata import fitted_card
from r0.modules.priors.lib.functional_analysis import functional_certificates
from r0.modules.priors.lib.functional_causal import FunctionalCausalRecipe
from r0.modules.priors.lib.functional_node_evaluation import evaluate_functional_nodes
from r0.modules.priors.lib.functional_prefix import FunctionalPrefix
from r0.modules.priors.lib.functional_target_calibration import (
    fit_target_calibration,
)
from r0.modules.priors.lib.joint_sampling_records import validate_path
from r0.modules.priors.lib.raw_draw_capture import RawDrawLedger, raw_transactions
from r0.modules.priors.lib.representatives import record_fingerprint
from r0.modules.priors.lib.sampled_functional_graph import SampledFunctionalGraph
from r0.modules.priors.lib.sampled_geometry_graph import SampledGeometryGraph
from r0.modules.priors.lib.sampled_parent_graph import SampledParentGraph
from r0.modules.priors.lib.sampled_topology_graph import SampledTopologyGraph
from r0.modules.priors.lib.task_output import complete_task
from r0.modules.priors.lib.training_task import TrainingTask


def functional_rng(seed: int, task_id: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(
        int(record_fingerprint(("functional-causal-v1", seed, task_id, keys)), 16)
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
    execution: InterventionExecution | None = None,
    population_recipe: FunctionalCausalRecipe | None = None,
) -> list[torch.Tensor]:
    rows = recipe.task.rows if rows is None else rows
    recipe.admit(evaluation_rows=rows)
    partitions: tuple[tuple[int, ...], ...] = ()
    if getattr(population_recipe or recipe, "population", None) == "support-only":
        if rows != recipe.task.rows:
            raise ValueError("support lifecycle requires its declared episode row count")
        order = tuple(map(int, rng("split", recipe.task.split).permutation(rows)))
        partitions = (order[: recipe.task.context_rows], order[recipe.task.context_rows :])
    return evaluate_functional_nodes(
        recipe.nodes,
        rows,
        rng,
        device,
        add=add,
        indices=indices,
        readouts=readouts,
        diagnostics=diagnostics,
        execution=execution,
        population_partitions=partitions,
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
    states = evaluate_functional_nodes(prefix.nodes, prefix.rows, rng, device, readouts=readouts)
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
        states = evaluate_functional_nodes(
            prefix.nodes,
            prefix.rows,
            lambda *keys: streams["calibration"],
            device,
            readouts=readouts,
            capture=ledger,
            owner=owner,
        )
        return PrefixEvaluation(tuple(states), tuple(readouts))


def generate_functional_causal(
    recipe: FunctionalCausalRecipe
    | SampledFunctionalGraph
    | SampledParentGraph
    | SampledTopologyGraph
    | SampledGeometryGraph,
    *,
    seed: int,
    task_id: int,
    device: str = "cpu",
    execution: InterventionExecution | None = None,
    calibration_recipe: FunctionalCausalRecipe | None = None,
) -> TrainingTask:
    construction = (
        recipe
        if isinstance(
            recipe,
            (
                SampledFunctionalGraph,
                SampledTopologyGraph,
                SampledGeometryGraph,
                SampledParentGraph,
            ),
        )
        else None
    )
    if construction is not None:
        construction.validate()
    recipe = (
        recipe.runtime
        if isinstance(
            recipe,
            (
                SampledFunctionalGraph,
                SampledTopologyGraph,
                SampledGeometryGraph,
                SampledParentGraph,
            ),
        )
        else recipe
    )
    if any(type(v) is not int or not 0 <= v < 2**63 for v in (seed, task_id)):
        raise ValueError("seed and task_id must be integers in [0, 2**63)")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; no implicit fallback")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; select CPU explicitly")

    def rng(*keys: object) -> np.random.Generator:
        def original(*parts: object) -> np.random.Generator:
            return functional_rng(seed, task_id, *parts)

        return execution.rng(original, *keys) if execution else original(*keys)

    values: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    functional_values(
        recipe,
        rng,
        device,
        readouts=values,
        diagnostics=diagnostics,
        execution=execution,
        population_recipe=calibration_recipe,
    )
    inputs = torch.stack([values[i][:, j] for i, j in recipe.observed], dim=1)
    scores = torch.stack([values[i][:, j] for i, j in recipe.target], dim=1)
    calibration = None
    calibration_source = calibration_recipe or recipe
    if calibration_source.target_calibration_rows:
        independent: list[torch.Tensor] = []
        functional_values(
            calibration_source,
            lambda *keys: functional_rng(seed, task_id, "target-calibration-v1", *keys),
            "cpu",
            readouts=independent,
            rows=calibration_source.target_calibration_rows,
        )
        raw = torch.stack([independent[i][:, j] for i, j in recipe.target], dim=1)
        calibration = fit_target_calibration(
            raw, record_fingerprint(([asdict(n) for n in calibration_source.nodes], recipe.target))
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
    card.update(fitted_card(recipe.version))
    from r0.modules.priors.lib.fitted_population_graph import PopulationFittedGraph

    if isinstance(recipe, PopulationFittedGraph):
        card.update(
            {
                "construction": "explicit-population-fit-v1",
                "fitting_population": recipe.population,
                "fitting_lifecycle": "explicit-fit-event-then-frozen-replay",
                "source_fidelity": "lifecycle-adaptation-not-full-source-reproduction",
            }
        )
    if recipe.version == "sampled-population-fitted-functional-causal-v1":
        card["construction"] = "sampled-parameter-population-fit-v1"
        card["sampling_provenance"] = "retained-parameter-and-population-custody-v1"
    from r0.modules.priors.lib.network_innovation_recipe import NetworkInnovationRecipe

    if isinstance(recipe, NetworkInnovationRecipe):
        assert recipe.plan is not None
        card["construction"] = "standalone-network-with-explicit-internal-innovations"
        card["network_innovations"] = {
            "profile": recipe.plan.profile,
            "source_scope": (
                "placement profile plus explicit scale/mean/importance laws; "
                "not full paper reproduction"
            ),
            "importance": (
                "multiplies outgoing weight columns and direct-input skips; "
                "gates only if explicitly selected"
            ),
            "masks": (
                "frozen structural masks; per-row innovations from independent keyed streams"
            ),
            "response": (
                "raw latent fixed-innovation or zero-innovation reference; "
                "not conditional mean or class probability"
            ),
            "block_nodes": recipe.block_nodes,
        }
        card["sampling_provenance"] = (
            "retained-statistics-stream-and-raw-records"
            if recipe.sampling is not None
            else "authored-noise-statistics"
        )
    if calibration is not None:
        card["normalization"] = "fixed independent calibration"
        card["calibration"] = asdict(calibration)
    if construction is not None:
        card["version"] = construction.version
        card["construction"] = (
            "sampled-parent-functions-v1"
            if isinstance(construction, SampledParentGraph)
            else "sampled-frozen-functions-v1"
        )
        card["sampling_provenance"] = "retained-four-stream-construction-v1"
        if isinstance(construction, (SampledTopologyGraph, SampledGeometryGraph)):
            card["construction"] = "sampled-topology-and-functions-v1"
            card["topology"] = asdict(construction.topology)
        if isinstance(construction, SampledGeometryGraph):
            card["construction"] = "sampled-geometry-and-functions-v1"
            card["geometry"] = {
                "latent_widths": construction.widths.latent,
                "converter_channels": construction.layout.converter_channels,
                "assignments": construction.observations.assignments,
                "converter_mode": (
                    "sampled-numerical-pipeline-v1"
                    if construction.request.version == "geometry-pipeline-request-v1"
                    else "identity-numerical-v1"
                ),
            }
    config = asdict(construction if construction is not None else recipe)
    source_config: dict[str, object] | None = None
    if construction is not None:
        source_config = construction_source_config(
            config,
            recipe.task,
            nested_graph=isinstance(construction, (SampledTopologyGraph, SampledGeometryGraph)),
        )
    return complete_task(
        recipe.task,
        config,
        card,
        inputs,
        scores,
        seed=seed,
        task_id=task_id,
        rng=rng,
        source_config=source_config,
    )
