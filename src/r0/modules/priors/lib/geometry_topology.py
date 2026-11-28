"""Strict topology variants and aggregate raw reservations for geometry graphs."""

from typing import Any, cast

import numpy as np

from r0.modules.priors.lib.cauchy_graph import CauchyGraph, CauchyGraphPrior, sample_cauchy_graph
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.fixed_parent_topology import (
    FixedParentTopology,
    RetainedParentTopology,
    retain_parent_topology,
)
from r0.modules.priors.lib.functional_causal_decoding import functional_array, functional_record
from r0.modules.priors.lib.raw_draw_records import RawDrawLimits
from r0.modules.priors.lib.structural_graph import StructuralGraph, sample_structural_graph
from r0.modules.priors.lib.structural_graph_prior import AttachmentGraphPrior, MotifGraphPrior

type GeometryTopologyPrior = (
    CauchyGraphPrior | AttachmentGraphPrior | MotifGraphPrior | FixedParentTopology
)
type GeometryTopology = CauchyGraph | StructuralGraph | RetainedParentTopology


def decode_geometry_topology(data: Any) -> GeometryTopologyPrior:
    if type(data) is not dict:
        raise ValueError("geometry topology prior must be an object")
    version = cast(dict[str, Any], data).get("version")
    if version == "fixed-parent-topology-r0-v1":
        raw = functional_record(data, FixedParentTopology)
        raw["parents"] = tuple(functional_array(row) for row in functional_array(raw["parents"]))
        return FixedParentTopology(**raw)
    if version == "cauchy-graph-prior-v1":
        return CauchyGraphPrior(**functional_record(data, CauchyGraphPrior))
    if version == "degree-attachment-dag-v1":
        return AttachmentGraphPrior(**functional_record(data, AttachmentGraphPrior))
    if version == "recursive-motif-dag-v1":
        raw = functional_record(data, MotifGraphPrior)
        for field in ("motifs", "rewrite_kinds"):
            if field in raw:
                raw[field] = functional_array(raw[field])
        return MotifGraphPrior(**raw)
    raise ValueError("unsupported geometry topology prior version")


def sample_geometry_topology(
    prior: GeometryTopologyPrior,
    *,
    rng: np.random.Generator,
    limits: CausalLimits,
    raw_limits: RawDrawLimits,
) -> GeometryTopology:
    if isinstance(prior, FixedParentTopology):
        return retain_parent_topology(prior, limits=limits)
    if isinstance(prior, CauchyGraphPrior):
        # Preserve the original Cauchy sampling schedule, internal limits and
        # serialized record exactly. Its full four-call cost is reserved below.
        return sample_cauchy_graph(prior, rng=rng, limits=limits)
    return sample_structural_graph(prior, rng=rng, limits=limits, raw_limits=raw_limits)


def _attachment_reservation(size: int, edges: int) -> tuple[int, int]:
    """Exact calls and categorical-support entries for one growing component."""
    h = min(edges, size - 1)
    later = size - 1 - h
    calls = h * (h + 1) // 2 + later * h
    # For predecessor count k<=h, sum supports k+(k-1)+...+1.
    entries = h * (h + 1) * (h + 2) // 6
    # For k>h, sum supports k+(k-1)+...+(k-h+1).
    entries += h * ((h + size) * later // 2) - later * h * (h - 1) // 2
    return calls, entries


def geometry_topology_raw_reservation(prior: GeometryTopologyPrior) -> tuple[int, int, int]:
    """Return maximum calls, scalar values and extra raw-parameter bytes.

    Scalar values and per-record headers are charged by GeometryGraphRequest.
    The extra term accounts for every retained attachment probability, which is
    record metadata rather than an emitted raw value. Motif rewrites reserve all
    three possible calls even when a particular attempted rewrite is a no-op.
    """
    if isinstance(prior, FixedParentTopology):
        return 0, 0, 256 * (prior.nodes + sum(map(len, prior.parents)) + 1)
    if isinstance(prior, CauchyGraphPrior):
        n = prior.nodes
        return 4, n * n + 2 * n + 1, 0
    if isinstance(prior, AttachmentGraphPrior):
        size, larger = divmod(prior.nodes, prior.components)
        calls_small, entries_small = _attachment_reservation(size, prior.edges_per_node)
        calls_large, entries_large = _attachment_reservation(size + 1, prior.edges_per_node)
        calls = (prior.components - larger) * calls_small + larger * calls_large
        entries = (prior.components - larger) * entries_small + larger * entries_large
        return calls, calls, 128 * entries
    calls = prior.components * (3**prior.depth - 1) // 2 + 3 * prior.rewrite_steps
    return calls, calls, 0
