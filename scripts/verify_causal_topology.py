"""Independent public-API checks for the partial causal-topology checkpoint.

Run with the intended checkout on PYTHONPATH and capture stdout as JSON.
Requires an ordinary Python invocation (assertions enabled). Statistical checks
supply bounded empirical evidence; they do not establish full issue #28 acceptance.
"""

import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import r0.modules.priors as priors_module
from r0.modules.priors import (
    CausalGraphPrior,
    CausalLimits,
    LinearRecipe,
    generate_task,
    recipe_from_dict,
    sample_causal_graph,
    sample_causal_recipe,
    sample_recipe,
)

torch.set_num_threads(1)
if not __debug__:
    raise RuntimeError("Run without -O: this verification procedure requires assertions")
source_directory = Path(priors_module.__file__).resolve().parent
actual_head = subprocess.check_output(
    ["git", "-C", str(source_directory), "rev-parse", "HEAD"], text=True
).strip()
report = {
    "head": actual_head,
    "original_reviewed_head": "b03dc24ea29372b33f3ae47980101d3da6cec3d1",
    "source_directory": str(source_directory),
    "scope": "partial topology checkpoint; full issue #28 incomplete",
    "checks": {},
}


def record(name, value):
    report["checks"][name] = value


def expect_reject(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    raise AssertionError("Expected rejection")


# Independent Bernoulli moments and edge covariance, with unequal partitions.
for family, layers, n, c in [("forward", (), 11, 3), ("layered", (2, 3, 1), 12, 2)]:
    prior = CausalGraphPrior(
        family=family, nodes=n, components=c, layer_widths=layers, edge_probability=0.35
    )
    allowed = []
    for comp in range(c):
        begin, end = n * comp // c, n * (comp + 1) // c
        layer_id = [i for i, w in enumerate(layers) for _ in range(w)]
        allowed += [
            (p, q)
            for p in range(begin, end)
            for q in range(p + 1, end)
            if family == "forward" or layer_id[q - begin] == layer_id[p - begin] + 1
        ]
    samples = []
    for seed in range(4000):
        graph = sample_causal_graph(prior, seed=seed, task_id=17)
        edges = {(p, q) for q, ps in enumerate(graph.parents) for p in ps}
        assert edges <= set(allowed)
        samples.append([int(edge in edges) for edge in allowed])
    a = np.array(samples)
    mean = float(a.mean())
    var = float(a.sum(1).var())
    expected = len(allowed) * 0.35 * 0.65
    assert abs(mean - 0.35) < 0.01
    assert abs(var - expected) < 0.4
    covariance = float(np.cov(a[:, 0], a[:, 1], ddof=0)[0, 1])
    assert abs(covariance) < 0.02
    for probability in (0, 1):
        g = sample_causal_graph(replace(prior, edge_probability=probability), seed=9, task_id=17)
        assert sum(map(len, g.parents)) == len(allowed) * probability
    record(
        family,
        {
            "draws": 4000,
            "eligible_edges": len(allowed),
            "empirical_p": mean,
            "count_variance": var,
            "binomial_variance": expected,
            "edge_covariance": covariance,
        },
    )

for p in (0, 0.5, 1):
    counts = Counter()
    for seed in range(6000):
        g = sample_causal_graph(
            CausalGraphPrior(
                family="redirection", nodes=8, components=2, redirection_probability=p
            ),
            seed=seed,
            task_id=19,
        )
        assert g.parents[0] == g.parents[4] == ()
        assert all(
            len(g.parents[i]) == 1 and 4 * (i // 4) <= g.parents[i][0] < i
            for i in (1, 2, 3, 5, 6, 7)
        )
        counts[g.parents[2][0]] += 1
        counts["node3root"] += g.parents[3] == (0,)
    expected = (1 + p) / 2
    expected3 = (1 + p + p * (1 + p) / 2) / 3
    assert abs(counts[0] / 6000 - expected) < 0.02
    assert abs(counts["node3root"] / 6000 - expected3) < 0.02
    record(
        f"redirection_{p}",
        {
            "draws": 6000,
            "node2_root": counts[0] / 6000,
            "expected": expected,
            "node3_root": counts["node3root"] / 6000,
            "expected3": expected3,
        },
    )

base = LinearRecipe(rows=32, context_rows=20, features=2, signal_features=2, active_features=1)
prior = CausalGraphPrior(nodes=4, edge_probability=1)
widths = (2, 3, 1, 4)
r = sample_causal_recipe(prior, task=base, seed=44, task_id=7, widths=widths)
for i, node in enumerate(r.nodes):
    assert len(node.bias) == widths[i]
    assert len(node.weights) == sum(widths[p] for p in node.parents)
    assert all(len(row) == widths[i] for row in node.weights)
assert not set(r.observed) & set(r.target)
assert len(set(r.observed + r.target)) == 3
rr = recipe_from_dict(json.loads(json.dumps(asdict(r))))
assert r == rr
assert r.topology == sample_causal_graph(prior, seed=44, task_id=7)
a = generate_task(r, seed=56, task_id=11)
b = generate_task(rr, seed=56, task_id=11)
assert torch.equal(a.inputs, b.inputs)
assert torch.equal(a.targets, b.targets)
changed = sample_causal_recipe(
    prior, task=replace(base, rows=64, context_rows=40, split=7), seed=44, task_id=7, widths=widths
)
assert r.nodes == changed.nodes
assert r.observed == changed.observed
assert r.target == changed.target
assert r.topology == changed.topology
large = generate_task(changed, seed=56, task_id=11)
assert torch.equal(a.inputs, large.inputs[:32])
assert torch.equal(a.targets, large.targets[:32])
record(
    "vector_json_row_independence",
    {
        "widths": widths,
        "observed": r.observed,
        "target": r.target,
        "replay": "bitwise equal",
        "rows32_vs64_prefix": "bitwise equal",
        "topology_and_mechanisms": "identical",
    },
)
# Admission measured with exact arithmetic, no author's tests or private imports.
params = sum(w * (2 + sum(widths[p] for p in r.nodes[i].parents)) for i, w in enumerate(widths))
limits = replace(CausalLimits(), parameters=params)
assert sample_causal_recipe(prior, task=base, seed=44, task_id=7, widths=widths, limits=limits)
record(
    "parameter_boundary",
    {
        "count": params,
        "exact_limit": "accepted",
        "one_less": expect_reject(
            lambda: sample_causal_recipe(
                prior,
                task=base,
                seed=44,
                task_id=7,
                widths=widths,
                limits=replace(limits, parameters=params - 1),
            )
        ),
    },
)
record(
    "channel_limits",
    expect_reject(
        lambda: sample_causal_recipe(
            prior,
            task=base,
            seed=44,
            task_id=7,
            widths=widths,
            limits=replace(limits, node_channels=3),
        )
    ),
)
record(
    "storage_limits",
    expect_reject(
        lambda: sample_causal_recipe(
            prior,
            task=base,
            seed=44,
            task_id=7,
            widths=widths,
            limits=replace(limits, working_bytes=10000),
        )
    ),
)
# Reject changed equation topology in saved JSON.
data = asdict(r)
data["nodes"][3]["parents"] = (0, 1)
data["nodes"][3]["weights"] = data["nodes"][3]["weights"][:5]
record(
    "tampered_equation_topology",
    expect_reject(lambda: recipe_from_dict(json.loads(json.dumps(data)))),
)
# Readout uniformity and Gaussian scaling over independently identified tasks.
readouts = Counter()
weights = []
biases = []
for seed in range(2000):
    one = sample_causal_recipe(prior, task=base, seed=seed, task_id=21, widths=widths)
    readouts[one.target[0]] += 1
    for node in one.nodes:
        if node.weights:
            weights.extend(np.array(node.weights).ravel() * math.sqrt(len(node.weights)))
        biases.extend(node.bias)
    assert len(set(one.observed + one.target)) == 3
assert max(abs(n - 200) for n in readouts.values()) < 60
assert abs(np.var(weights) - 1) < 0.02
assert abs(np.mean(weights)) < 0.02
assert abs(np.var(biases) - 0.0625) < 0.003
record(
    "readout_and_weights",
    {
        "draws": 2000,
        "target_channel_counts": {str(k): v for k, v in sorted(readouts.items())},
        "standardized_weight_mean": float(np.mean(weights)),
        "standardized_weight_variance": float(np.var(weights)),
        "bias_variance": float(np.var(biases)),
    },
)
for family in ("forward", "layered", "redirection"):
    for seed in range(100):
        p = sample_recipe(family, profile="causal-affine-v1", task=base, seed=seed, task_id=0)
        p2 = sample_recipe(
            family,
            profile="causal-affine-v1",
            task=replace(base, rows=64, context_rows=40),
            seed=seed,
            task_id=0,
        )
        assert p.nodes == p2.nodes
        assert p.topology == p2.topology
        assert p.observed == p2.observed
        assert p.target == p2.target
record("profile_row_independence", "300 paired draws passed")
# Classification adapters preserve disjoint multi-channel readouts and public episode boundary.
classification = {}
for kind, classes in [("binary", 2), ("multiclass", 5)]:
    task = replace(base, task_type=kind, classes=classes)
    for seed in range(50):
        recipe = sample_causal_recipe(prior, task=task, seed=seed, task_id=31, widths=widths)
        assert len(recipe.target) == (5 if kind == "multiclass" else 1)
        assert not set(recipe.target) & set(recipe.observed)
        out = generate_task(recipe, seed=seed, task_id=32)
        assert out.probabilities.shape == (32, classes)
        assert torch.allclose(out.probabilities.sum(1), torch.ones(32, dtype=torch.float64))
        assert set(out.targets.tolist()) <= set(range(classes))
        episode = out.episode()
        assert len(episode.context_targets) == 20
        assert len(episode.query_targets) == 12
    classification[kind] = "50 draws passed"
report["checks"]["classification"] = classification
print(json.dumps(report, indent=2))
