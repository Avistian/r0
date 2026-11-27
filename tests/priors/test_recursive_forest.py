"""Literal independent traversal, source variants, replay and resource boundaries."""

from __future__ import annotations

# ruff: noqa: B023 -- recursive reference is fully evaluated before loop variables change.
import copy
import json
from dataclasses import asdict, replace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from r0.modules import priors as p


def oracle(plan: p.RecursiveForestPlan, rng: np.random.Generator):
    """Small recursive reference; does not call the iterative builder or raw ledger."""
    forests: list[list[list[list[Any]]]] = []
    for _ in range(plan.populations):
        trees: list[list[list[Any]]] = []
        for _ in range(plan.estimators):
            start = (
                int(rng.integers(plan.classes, size=1)[0]) if plan.task == "classification" else 0
            )
            offset = start if plan.class_assignment == "random-start-cycle" else 0
            nodes: list[list[Any]] = []
            leaves = 0

            def visit(depth: int, bounds: tuple[tuple[float, float], ...]):
                nonlocal leaves
                index = len(nodes)
                dim = int(rng.integers(plan.input_width, size=1)[0])
                low, high = bounds[dim]
                threshold = low + (high - low) * float(rng.random(size=1)[0])
                decision = float(rng.random(size=1)[0]) if depth < plan.max_depth else -1.0
                split = depth < plan.min_depth or decision >= plan.terminal_probability
                nodes.append([dim, threshold, (low, high), depth, -1, -1, (), None])
                if split:
                    left = list(bounds)
                    left[dim] = (low, threshold)
                    right = list(bounds)
                    right[dim] = (threshold, high)
                    nodes[index][4] = visit(depth + 1, tuple(left))
                    nodes[index][5] = visit(depth + 1, tuple(right))
                elif plan.task == "classification":
                    nodes[index][7] = (offset + leaves) % plan.classes
                    leaves += 1
                else:
                    z = rng.normal(size=plan.output_width)
                    nodes[index][6] = tuple(
                        float(m + s * v)
                        for m, s, v in zip(plan.leaf_mean, plan.leaf_scale, z, strict=True)
                    )
                return index

            visit(0, plan.intervals)
            trees.append(nodes)
        forests.append(trees)
    return forests


@pytest.mark.parametrize(
    "engine",
    [np.random.PCG64, np.random.PCG64DXSM, np.random.Philox, np.random.SFC64, np.random.MT19937],
)
@pytest.mark.parametrize(
    ("source", "task"),
    [
        ("mitra-prose-v1", "regression"),
        ("mitra-prose-v1", "classification"),
        ("mitra-algorithm4-v1", "classification"),
    ],
)
def test_source_tree_literal_reference_and_replay(
    engine: type[np.random.BitGenerator], source: str, task: str
):
    plan = p.RecursiveForestPlan(
        input_width=1,
        intervals=((-2.0, 3.0),),
        estimators=2,
        max_depth=3,
        terminal_probability=0.6,
        source=source,
    )
    if task == "classification":
        plan = replace(
            plan,
            task=task,
            classes=3,
            leaf_mean=(),
            leaf_scale=(),
            tie_break="smallest",
            class_assignment="random-start-cycle"
            if source == "mitra-prose-v1"
            else "zero-start-cycle",
        )
    rng = np.random.Generator(engine(23))
    reference = np.random.Generator(engine(23))
    expected = oracle(plan, reference)
    sample = p.sample_recursive_forest(plan, rng=rng)
    for forest, trees in zip(sample.forests, expected, strict=True):
        for tree, nodes in zip(forest.trees, trees, strict=True):
            assert [list(asdict(n).values()) for n in tree.nodes] == nodes
    assert json.dumps(rng.bit_generator.state, default=lambda x: x.tolist()) == json.dumps(
        reference.bit_generator.state, default=lambda x: x.tolist()
    )
    assert p.sampled_recursive_forest_from_dict(json.loads(json.dumps(asdict(sample)))) == sample


def test_inclusive_threshold_and_no_float32_cast():
    nodes = (
        p.RecursiveTreeNode(0, 0.0, (-1.0, 1.0), 0, 1, 2),
        p.RecursiveTreeNode(0, -0.5, (-1.0, 0.0), 1, value=(-3.0, 5.0)),
        p.RecursiveTreeNode(0, 0.5, (0.0, 1.0), 1, value=(7.0, -2.0)),
    )
    forest = p.RecursiveForest(1, (p.RecursiveTree(nodes),), output_width=2)
    x = np.array([[-1e-100], [0.0], [1e-100], [20.0]], dtype=np.float64)
    np.testing.assert_array_equal(
        p.evaluate_recursive_forest(forest, x), [[-3.0, 5.0], [-3.0, 5.0], [7.0, -2.0], [7.0, -2.0]]
    )
    assert p.inspect_recursive_forest(forest, x) == (((0, 1),), ((0, 1),), ((0, 2),), ((0, 2),))
    assert p.recursive_forest_from_dict(asdict(forest)) == forest


@pytest.mark.parametrize(("tie", "expected"), [("smallest", 0), ("largest", 1), ("first-tree", 1)])
def test_actual_class_majority_ties(tie: str, expected: int):
    trees = tuple(
        p.RecursiveTree((p.RecursiveTreeNode(0, 0.0, (-1.0, 1.0), 0, label=i),)) for i in (1, 0)
    )
    forest = p.RecursiveForest(1, trees, "classification", 1, 2, tie)
    result = p.evaluate_recursive_forest(forest, np.zeros((3, 1)))
    assert result.dtype == np.dtype("int64")
    np.testing.assert_array_equal(result, [expected] * 3)


def test_minimum_maximum_and_terminal_extremes():
    base = p.RecursiveForestPlan(input_width=1, intervals=((-1.0, 1.0),), estimators=1, max_depth=4)
    full = p.sample_recursive_forest(
        replace(base, terminal_probability=0.0), rng=np.random.default_rng(3)
    )
    early = p.sample_recursive_forest(
        replace(base, terminal_probability=1.0), rng=np.random.default_rng(3)
    )
    assert len(full.forests[0].trees[0].nodes) == 31
    assert len(early.forests[0].trees[0].nodes) == 3
    assert max(n.depth for n in full.forests[0].trees[0].nodes) == 4


def test_per_row_forest_custody_and_vector_weighted_extensions():
    plan = p.RecursiveForestPlan(
        input_width=2,
        intervals=((-2.0, 2.0), (-1.0, 3.0)),
        source="r0-extension-v1",
        lifecycle="fresh-forest-per-row",
        populations=3,
        estimators=2,
        max_depth=2,
        split_probabilities=(0.0, 1.0),
        leaf_mean=(1.0, -2.0),
        leaf_scale=(0.5, 2.0),
    )
    sample = p.sample_recursive_forest(plan, rng=np.random.default_rng(37))
    assert len(sample.forests) == 3
    assert all(n.feature == 1 for f in sample.forests for t in f.trees for n in t.nodes)
    x = np.zeros((3, 2))
    expected = np.concatenate(
        [p.evaluate_recursive_forest(f, x[i : i + 1]) for i, f in enumerate(sample.forests)]
    )
    np.testing.assert_array_equal(p.evaluate_recursive_forest(sample, x), expected)
    assert p.sampled_recursive_forest_from_dict(asdict(sample)) == sample
    with pytest.raises(ValueError, match="row count"):
        p.evaluate_recursive_forest(sample, x[:2])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nodes", 1),
        ("node_channels", 1),
        ("total_channels", 1),
        ("parameters", 1),
        ("working_bytes", 1),
    ],
)
def test_caps_before_rng_snapshot(field: str, value: int):
    limits = replace(p.CausalLimits(), **{field: value})
    rng = np.random.default_rng(9)
    before = copy.deepcopy(rng.bit_generator.state)
    with (
        patch(
            "r0.modules.priors.lib.recursive_forest_sampling.snapshot_generator",
            side_effect=AssertionError("snapshot before admission"),
        ),
        pytest.raises(ValueError, match="limit"),
    ):
        p.sample_recursive_forest(p.RecursiveForestPlan(), rng=rng, limits=limits)
    assert rng.bit_generator.state == before


def test_raw_budget_and_late_failure_rollback():
    plan = p.RecursiveForestPlan(
        source="r0-extension-v1",
        min_depth=0,
        max_depth=0,
        estimators=1,
        leaf_mean=(1.7e308,),
        leaf_scale=(1.7e308,),
    )
    rng = np.random.default_rng(1)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="node"):
        p.sample_recursive_forest(plan, rng=rng)
    assert rng.bit_generator.state == before
    with pytest.raises(ValueError, match="raw limit"):
        p.sample_recursive_forest(plan, rng=rng, raw_limits=p.RawDrawLimits(draws=1))
    assert rng.bit_generator.state == before


def test_evaluation_rejects_rows_before_finite_mask():
    sample = p.sample_recursive_forest(
        p.RecursiveForestPlan(estimators=1, max_depth=1), rng=np.random.default_rng(8)
    )
    x = np.zeros((1000, 2))
    with (
        patch.object(np, "isfinite", side_effect=AssertionError("finite mask before admission")),
        pytest.raises(ValueError, match="workspace|storage"),
    ):
        p.evaluate_recursive_forest(sample.forests[0], x, limits=p.CausalLimits(working_bytes=1000))


@pytest.mark.parametrize("field", ["threshold", "raw", "final", "label", "version"])
def test_archive_tamper_rejects(field: str):
    sample = p.sample_recursive_forest(
        p.RecursiveForestPlan(max_depth=1, estimators=1), rng=np.random.default_rng(13)
    )
    data = json.loads(json.dumps(asdict(sample)))
    if field == "threshold":
        data["forests"][0]["trees"][0]["nodes"][0]["threshold"] = 0.0
    elif field == "raw":
        data["raw"][0]["values"][0] = 99
    elif field == "final":
        data["final"]["digest"] = "0" * 64
    elif field == "label":
        data["forests"][0]["trees"][0]["nodes"][1]["value"][0] = False
    else:
        data["version"] = "other"
    with pytest.raises(ValueError, match="recursive"):
        p.sampled_recursive_forest_from_dict(data)


def test_decoder_caps_before_replay():
    sample = p.sample_recursive_forest(
        p.RecursiveForestPlan(max_depth=1, estimators=1), rng=np.random.default_rng(13)
    )
    with (
        patch.object(type(sample.initial), "restore", side_effect=AssertionError("early replay")),
        pytest.raises(ValueError, match="limit"),
    ):
        p.sampled_recursive_forest_from_dict(asdict(sample), limits=p.CausalLimits(nodes=1))


def test_iid_extension_uses_no_unused_class_start():
    plan = p.RecursiveForestPlan(
        source="r0-extension-v1",
        task="classification",
        classes=4,
        leaf_mean=(),
        leaf_scale=(),
        class_assignment="iid",
        tie_break="largest",
        estimators=1,
        max_depth=2,
    )
    sample = p.sample_recursive_forest(plan, rng=np.random.default_rng(14))
    assert not any(r.role == "class-start" for r in sample.raw)
    assert any(r.role == "leaf-label" for r in sample.raw)
    assert p.sampled_recursive_forest_from_dict(asdict(sample)) == sample


@pytest.mark.parametrize(
    "change",
    [
        dict(min_depth=5),
        dict(terminal_probability=1.1),
        dict(source="mitra-prose-v1", split_probabilities=(0.5, 0.5)),
        dict(source="mitra-prose-v1", leaf_mean=(0.0, 1.0), leaf_scale=(1.0, 1.0)),
        dict(lifecycle="frozen-function", populations=2),
        dict(classes=3),
        dict(intervals=((0.0, 0.0), (0.0, 1.0))),
    ],
)
def test_explicit_incompatible_controls_reject(change: dict[str, Any]):
    with pytest.raises(ValueError, match="recursive|source|regression"):
        p.RecursiveForestPlan(**change)


def test_declared_limits_remain_active_during_evaluation():
    plan = p.RecursiveForestPlan(estimators=1, max_depth=1)
    bound = plan.admit(p.CausalLimits(), p.RawDrawLimits())["estimated_working_bytes"]
    sample = p.sample_recursive_forest(
        plan, rng=np.random.default_rng(4), limits=p.CausalLimits(working_bytes=bound + 1000)
    )
    with (
        patch.object(np, "isfinite", side_effect=AssertionError("before declared cap")),
        pytest.raises(ValueError, match="workspace"),
    ):
        p.evaluate_recursive_forest(sample, np.zeros((10, 2)))


def test_invalid_interval_and_boolean_leaf_reject_as_authored():
    sample = p.sample_recursive_forest(
        p.RecursiveForestPlan(estimators=1, max_depth=1), rng=np.random.default_rng(4)
    )
    raw = asdict(sample.forests[0])
    raw["trees"][0]["nodes"][0]["threshold"] = True
    with pytest.raises(ValueError, match="recursive"):
        p.recursive_forest_from_dict(raw)


def test_sampled_record_type_exact_leaf_rejection():
    plan = p.RecursiveForestPlan(estimators=1, max_depth=1, leaf_mean=(1.0,), leaf_scale=(0.0,))
    sample = p.sample_recursive_forest(plan, rng=np.random.default_rng(4))
    raw = json.loads(json.dumps(asdict(sample)))
    raw["forests"][0]["trees"][0]["nodes"][1]["value"] = [1]
    with pytest.raises(ValueError, match="recursive"):
        p.sampled_recursive_forest_from_dict(raw)


def test_actual_geometry_rejects_before_snapshot_restore():
    sample = p.sample_recursive_forest(
        p.RecursiveForestPlan(estimators=1, max_depth=1), rng=np.random.default_rng(4)
    )
    other = replace(sample.forests[0], input_width=3)
    with (
        patch.object(type(sample.initial), "restore", side_effect=AssertionError("early replay")),
        pytest.raises(ValueError, match="geometry"),
    ):
        replace(sample, forests=(other,))


def test_worst_workspace_boundary_before_snapshot():
    plan = p.RecursiveForestPlan(estimators=1, max_depth=1)
    bound = plan.admit(p.CausalLimits(), p.RawDrawLimits())["estimated_working_bytes"]
    admitted = p.sample_recursive_forest(
        plan, rng=np.random.default_rng(4), limits=p.CausalLimits(working_bytes=bound)
    )
    assert p.sampled_recursive_forest_from_dict(asdict(admitted)) == admitted
    with (
        patch(
            "r0.modules.priors.lib.recursive_forest_sampling.snapshot_generator",
            side_effect=AssertionError("early snapshot"),
        ),
        pytest.raises(ValueError, match="workspace"),
    ):
        p.sample_recursive_forest(
            plan, rng=np.random.default_rng(4), limits=p.CausalLimits(working_bytes=bound - 1)
        )
