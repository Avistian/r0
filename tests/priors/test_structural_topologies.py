"""Retained structural topology draws and source-distinct growth mechanisms."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from r0.modules import priors


def test_preferential_growth_matches_direct_degree_weight_oracle() -> None:
    prior = priors.AttachmentGraphPrior(
        nodes=9, components=2, edges_per_node=2, attractiveness=0.7, power=1.3
    )
    rng = np.random.default_rng(41)
    oracle = copy.deepcopy(rng)
    graph = priors.sample_structural_graph(prior, rng=rng)
    expected: list[tuple[int, ...]] = []
    degrees = [0] * 9
    for start, stop in ((0, 4), (4, 9)):
        for node in range(start, stop):
            candidates = list(range(start, node))
            selected: list[int] = []
            for _ in range(min(2, len(candidates))):
                weights = np.array([(degrees[i] + 0.7) ** 1.3 for i in candidates])
                index = int(oracle.choice(len(candidates), p=weights / weights.sum()))
                parent = candidates.pop(index)
                degrees[parent] += 1
                degrees[node] += 1
                selected.append(parent)
            expected.append(tuple(sorted(selected)))
    assert graph.parents == tuple(expected)
    np.testing.assert_array_equal(rng.random(8), oracle.random(8))
    graph.validate()
    assert graph.replay(np.random.default_rng(41)) == graph
    broken = replace(graph, parents=((),) * 9)
    with pytest.raises(ValueError, match="differs"):
        broken.validate()


@pytest.mark.parametrize(
    ("motif", "expected"),
    [
        ("chain", ((), (0,), (1,))),
        ("fork", ((), (0,), (0,))),
        ("collider", ((), (), (0, 1))),
    ],
)
def test_one_level_motifs_have_exact_ports(
    motif: str, expected: tuple[tuple[int, ...], ...]
) -> None:
    graph = priors.sample_structural_graph(
        priors.MotifGraphPrior(depth=1, motifs=(motif,)), rng=np.random.default_rng(11)
    )
    assert graph.parents == expected
    assert graph.replay(np.random.default_rng(11)) == graph


def test_recursive_motifs_and_rewrites_retain_components_and_failure_rollback() -> None:
    prior = priors.MotifGraphPrior(depth=2, components=2, rewrite_steps=8)
    rng = np.random.default_rng(20)
    graph = priors.sample_structural_graph(prior, rng=rng)
    assert len(graph.parents) == 18
    assert len(graph.rewrites) == 8
    assert all(parent < node for node, parents in enumerate(graph.parents) for parent in parents)
    assert all(
        parent // 9 == node // 9 for node, parents in enumerate(graph.parents) for parent in parents
    )
    assert graph.replay(np.random.default_rng(20)) == graph
    before = copy.deepcopy(rng)
    with pytest.raises(ValueError, match="limit"):
        priors.sample_structural_graph(prior, rng=rng, limits=priors.CausalLimits(parameters=10))
    np.testing.assert_array_equal(rng.random(8), before.random(8))


@pytest.mark.parametrize("engine", ["PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"])
@pytest.mark.parametrize("family", ["attachment", "motif"])
def test_structural_replay_and_raw_limit_rollback_across_engines(engine: str, family: str) -> None:
    prior = (
        priors.AttachmentGraphPrior(nodes=7, edges_per_node=2)
        if family == "attachment"
        else priors.MotifGraphPrior(depth=2, rewrite_steps=5)
    )
    rng = np.random.Generator(getattr(np.random, engine)(91))
    rng.random(5)
    initial = copy.deepcopy(rng)
    graph = priors.sample_structural_graph(prior, rng=rng)
    replay = copy.deepcopy(initial)
    assert graph.replay(replay) == graph
    np.testing.assert_array_equal(rng.random(8), replay.random(8))
    corrupt = replace(graph, raw=(replace(graph.raw[0], state_before="0" * 64), *graph.raw[1:]))
    failed = copy.deepcopy(initial)
    control = copy.deepcopy(initial)
    with pytest.raises(ValueError, match="state"):
        corrupt.replay(failed)
    np.testing.assert_array_equal(failed.random(8), control.random(8))
    failed, control = copy.deepcopy(initial), copy.deepcopy(initial)
    with pytest.raises(ValueError, match="raw draw.*limit"):
        priors.sample_structural_graph(prior, rng=failed, raw_limits=priors.RawDrawLimits(draws=1))
    np.testing.assert_array_equal(failed.random(8), control.random(8))


@pytest.mark.parametrize("kind", ["redirect", "path", "bypass"])
def test_rewrite_receipts_reconstruct_every_changed_edge_and_preserve_order(kind: str) -> None:
    applied = 0
    for seed in range(8):
        prior = priors.MotifGraphPrior(depth=2, rewrite_steps=8, rewrite_kinds=(kind,))
        base = priors.sample_structural_graph(
            replace(prior, rewrite_steps=0), rng=np.random.default_rng(seed)
        )
        graph = priors.sample_structural_graph(prior, rng=np.random.default_rng(seed))
        edges = {
            (parent, node) for node, incoming in enumerate(base.parents) for parent in incoming
        }
        for receipt in graph.rewrites:
            assert set(receipt.removed) <= edges
            assert not set(receipt.added) & edges
            if receipt.reason == "applied":
                applied += 1
                assert receipt.removed
                if kind in ("redirect", "path"):
                    source, target, replacement = receipt.subject
                    assert receipt.removed == ((source, target),)
                    if kind == "redirect":
                        assert receipt.added == ((replacement, target),)
                        assert replacement < target
                    else:
                        assert receipt.added == tuple(
                            sorted(((source, replacement), (replacement, target)))
                        )
                        assert source < replacement < target
                else:
                    node = receipt.subject[0]
                    incoming = {a for a, b in edges if b == node}
                    outgoing = {b for a, b in edges if a == node}
                    assert set(receipt.removed) == {
                        (a, b) for a, b in edges if a == node or b == node
                    }
                    assert (
                        set(receipt.added) == {(a, b) for a in incoming for b in outgoing} - edges
                    )
            else:
                assert not receipt.removed
                assert not receipt.added
            edges.difference_update(receipt.removed)
            edges.update(receipt.added)
            assert all(a < b for a, b in edges)
        assert edges == {
            (parent, node) for node, incoming in enumerate(graph.parents) for parent in incoming
        }
    assert applied > 0


def test_empty_topologies_and_explicit_growth_kernel_variants() -> None:
    for prior in (
        priors.AttachmentGraphPrior(nodes=3, components=3),
        priors.MotifGraphPrior(depth=0, components=3),
    ):
        rng = np.random.default_rng(9)
        control = copy.deepcopy(rng)
        graph = priors.sample_structural_graph(prior, rng=rng)
        assert graph.parents == ((), (), ())
        assert graph.raw == ()
        np.testing.assert_array_equal(rng.random(8), control.random(8))
    for power in (0.0, 0.5, 1.0, 2.0):
        graph = priors.sample_structural_graph(
            priors.AttachmentGraphPrior(nodes=8, power=power), rng=np.random.default_rng(17)
        )
        assert all(len(incoming) == 1 for incoming in graph.parents[1:])
