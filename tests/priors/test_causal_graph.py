"""Topology contracts at the public sampler, using independently known graphs."""

from dataclasses import replace

from r0.modules.priors import CausalGraphPrior, sample_causal_graph


def test_forward_graph_keeps_disconnected_components_and_extreme_probabilities() -> None:
    prior = CausalGraphPrior(nodes=6, components=2, edge_probability=1.0)
    full = sample_causal_graph(prior, seed=28, task_id=0)
    # Two disjoint three-vertex complete DAGs in parent-before-child order.
    assert full.parents == ((), (0,), (0, 1), (), (3,), (3, 4))
    assert full.prior == prior
    empty = sample_causal_graph(replace(prior, edge_probability=0), seed=28, task_id=0)
    assert empty.parents == ((),) * 6
    sparse = replace(prior, nodes=18, components=1, edge_probability=0.3)
    draw = sample_causal_graph(sparse, seed=28, task_id=1)
    assert draw == sample_causal_graph(sparse, seed=28, task_id=1)
    assert draw.parents != sample_causal_graph(sparse, seed=28, task_id=2).parents
