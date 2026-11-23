"""Scalar construction checks independent of vectorized representative samplers."""

import json
import math
from dataclasses import dataclass
from statistics import pvariance
from typing import Any, cast

from r0.modules.priors import GeneratedTask


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    severity: str
    detail: str
    proposal_id: str | None = None


def _close(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-10 + 1e-10 * abs(b)


def _latent(task: GeneratedTask, parameters: dict[str, Any], trace: dict[str, Any]) -> list[float]:
    rows = [[float(cast(float, v)) for v in row.values] for row in task.source_rows]
    family = task.proposal.family
    if family == "linear":
        return [sum(v * w for v, w in zip(row, parameters["weights"], strict=True)) for row in rows]
    if family == "interaction":
        degree, rule = parameters["degree"], parameters["rule"]
        if rule not in ("product", "parity", "gated"):
            raise ValueError("unknown interaction rule")
        return [
            math.prod(row[:degree])
            if rule == "product"
            else float(sum(v > 0 for v in row[:degree]) % 2)
            if rule == "parity"
            else math.prod(row[1:degree])
            if row[0] > 0
            else -1.0
            for row in rows
        ]
    if family == "mlp":
        result: list[float] = []
        for row in rows:
            values = row
            for layer, weights in enumerate(parameters["weights"]):
                values = [
                    sum(v * w for v, w in zip(values, column, strict=True))
                    for column in zip(*weights, strict=True)
                ]
                if layer < len(parameters["weights"]) - 1:
                    values = [
                        math.tanh(v) if parameters["activation"] == "tanh" else max(0.0, v)
                        for v in values
                    ]
            result.append(values[0])
        return result
    if family == "tree":
        result = []
        for row, recorded_leaf in zip(rows, trace["leaf_indices"], strict=True):
            node = 0
            for _ in range(parameters["depth"]):
                right = row[int(parameters["split_features"][node])] > float(
                    parameters["thresholds"][node]
                )
                node = 2 * node + 1 + right
            leaf = node - len(parameters["split_features"])
            if type(recorded_leaf) is not int or recorded_leaf != leaf:
                raise ValueError("tree leaf trace disagrees with independently traversed path")
            result.append(parameters["leaves"][leaf])
        return result
    if family == "scm":
        result = []
        for row, noises, recorded in zip(rows, trace["node_noise"], trace["nodes"], strict=True):
            nodes: list[float] = []
            for i, parents in enumerate(cast(list[list[int]], parameters["parents"])):
                value = noises[i]
                if parents:
                    value += math.tanh(
                        sum(
                            nodes[parent] * weight
                            for parent, weight in zip(
                                parents, parameters["weights"][i], strict=True
                            )
                        )
                    )
                nodes.append(value)
            if any(not _close(a, b) for a, b in zip(nodes, recorded, strict=True)):
                raise ValueError("SCM node construction disagrees with retained trace")
            for column, node in enumerate(cast(list[int], parameters["observed_nodes"])):
                if not _close(row[column], nodes[node]):
                    raise ValueError("SCM observed feature does not match node")
            result.append(nodes[int(parameters["target_node"])])
        return result
    # Verify the full GP covariance factor and joint function draw independently.
    knots, factor, z = parameters["knots"], parameters["cholesky"], parameters["standard_normal"]
    values = [sum(a * b for a, b in zip(row, z, strict=True)) for row in factor]
    for i, x in enumerate(knots):
        for j, y in enumerate(knots):
            expected = parameters["signal_variance"] * math.exp(
                -0.5 * ((x - y) / parameters["length_scale"]) ** 2
            )
            if i == j:
                expected += parameters["diagonal_component_variance"]
            actual = sum(a * b for a, b in zip(factor[i], factor[j], strict=True))
            if not _close(actual, expected):
                raise ValueError("GP factor disagrees with declared covariance")
    if any(not _close(a, b) for a, b in zip(values, parameters["values"], strict=True)):
        raise ValueError("GP function draw disagrees with whitened innovations")
    for row, index in zip(rows, trace["support_indices"], strict=True):
        if row != [knots[index]]:
            raise ValueError("GP observation is not its recorded support coordinate")
    return [values[i] for i in trace["support_indices"]]


def inspect_task(task: GeneratedTask) -> tuple[tuple[Finding, ...], dict[str, object]]:
    findings: list[Finding] = []
    descriptors: dict[str, object] = {}

    def fail(code: str, detail: str) -> None:
        findings.append(Finding(code, "error", detail, task.proposal.id))

    try:
        mechanism: dict[str, Any] = json.loads(task.mechanism_json)
        trace: dict[str, Any] = json.loads(task.trace_json)
        parameters = mechanism["parameters"]
        if (
            mechanism["family"] != task.proposal.family
            or mechanism["version"] != task.proposal.version
        ):
            fail("RECORD_INVALID", "mechanism family/version disagrees with proposal")
        if (
            len(task.source_rows) != task.proposal.rows
            or len(task.source_targets) != task.proposal.rows
        ):
            fail("RECORD_INVALID", "source shape disagrees with proposal")
        sources = {r.id: (r, y) for r, y in zip(task.source_rows, task.source_targets, strict=True)}
        if len(sources) != len(task.source_rows):
            fail("RECORD_INVALID", "source row identities are not unique")
        episode = task.episode
        proposal = task.proposal
        expected_streams = tuple(
            (name, proposal.stream(name)[1])
            for name in ("mechanism", "observations", "noise", "split")
        )
        if task.streams != expected_streams:
            fail("STREAM_INTEGRITY", "stream metadata disagrees with declared proposal")
        if (
            episode.task_type != proposal.task_type
            or mechanism["target_type"] != proposal.task_type
        ):
            fail("RECORD_INVALID", "target type disagrees with proposal")
        expected_features = 1 if proposal.family == "gp" else proposal.features
        if len(episode.features) != expected_features:
            fail("RECORD_INVALID", "feature count disagrees with representative proposal")
        split_rng, _ = proposal.stream("split")
        order = [int(i) for i in split_rng.permutation(proposal.rows)]
        expected_context = tuple(task.source_rows[i] for i in order[: proposal.context_rows])
        expected_queries = tuple(task.source_rows[i] for i in order[proposal.context_rows :])
        if episode.context != expected_context or episode.queries != expected_queries:
            fail("PARTITION_INTEGRITY", "partition disagrees with declared split stream")
        combined = episode.context + episode.queries
        labels = episode.context_targets + episode.query_targets
        if set(sources) != {r.id for r in combined} or any(
            sources.get(row.id) != (row, y) for row, y in zip(combined, labels, strict=True)
        ):
            fail("SOURCE_EPISODE_MISMATCH", "source observations and episode partitions disagree")
        expected_latent = _latent(task, parameters, trace)
        if any(not _close(a, b) for a, b in zip(expected_latent, trace["latent"], strict=True)):
            fail("LATENT_INTEGRITY", "latent values disagree with independent rule calculation")
        expected_targets: list[float | str] = []
        if task.proposal.task_type == "regression":
            for value, noise in zip(expected_latent, trace["target_noise"], strict=True):
                if task.proposal.noise == "none" and noise != 0:
                    fail("NOISE_INTEGRITY", "no-added-noise task has nonzero observation noise")
                expected_targets.append(value + noise)
        else:
            adapter = mechanism["target_adapter"]
            class_count = 2 if task.proposal.task_type == "binary" else 3
            if (
                len(episode.classes) != class_count
                or adapter["thresholds"] != ([0.0] if class_count == 2 else [-0.5, 0.5])
                or sorted(adapter["mapping"]) != list(range(class_count))
                or any(type(v) is not int for v in adapter["mapping"])
            ):
                raise ValueError("invalid declared target adapter/vocabulary")
            if any(
                len(trace[key]) != task.proposal.rows
                for key in ("base_class", "replacement_event", "replacement_class", "label_changed")
            ):
                raise ValueError("label trace dimensions disagree with proposal")
            for i, value in enumerate(expected_latent):
                base = adapter["mapping"][sum(value >= t for t in adapter["thresholds"])]
                event, replacement = trace["replacement_event"][i], trace["replacement_class"][i]
                if (
                    type(event) is not bool
                    or type(replacement) is not int
                    or not 0 <= replacement < class_count
                    or type(trace["base_class"][i]) is not int
                    or type(trace["label_changed"][i]) is not bool
                ):
                    raise ValueError("invalid discrete label-randomization trace")
                final = replacement if event else base
                if (
                    trace["base_class"][i] != base
                    or trace["label_changed"][i] != (final != base)
                    or (task.proposal.noise == "none" and event)
                ):
                    fail("NOISE_INTEGRITY", f"label-randomization trace inconsistent at row {i}")
                expected_targets.append(episode.classes[int(final)])
        for i, (expected, actual) in enumerate(
            zip(expected_targets, task.source_targets, strict=True)
        ):
            valid = (
                expected == actual if isinstance(expected, str) else _close(expected, float(actual))
            )
            if not valid:
                fail("LABEL_INTEGRITY", f"row {i}: expected {expected!r}, observed {actual!r}")
        relevant = len(parameters["relevant"])
        latent_variance = pvariance(expected_latent)
        noise_variance = (
            pvariance(trace["target_noise"]) if task.proposal.task_type == "regression" else None
        )
        descriptors = {
            "features": len(episode.features),
            "relevant_features": relevant,
            "nuisance_features": len(episode.features) - relevant,
            "sparsity_band": "at_most_two_relevant" if relevant <= 2 else "more_than_two_relevant",
            "latent_sample_variance": latent_variance,
            "added_noise_sample_variance": noise_variance,
            "signal_noise_variance_ratio": latent_variance / noise_variance
            if noise_variance
            else None,
            "signal_noise_ratio_reason": None
            if noise_variance
            else "zero noise variance or classification",
            "class_counts": {c: task.source_targets.count(c) for c in episode.classes},
            "missing_values": sum(v is None for r in task.source_rows for v in r.values),
            "downstream_usefulness": "NOT_MEASURED",
        }
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        fail("RECORD_INVALID", f"cannot verify construction: {error}")
    return tuple(findings), descriptors
