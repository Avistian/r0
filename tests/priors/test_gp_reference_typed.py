"""Typed targets and captures preserve categories and full row-domain semantics."""

import hashlib
import json
from dataclasses import asdict, replace
from typing import Literal

import numpy as np
import pytest
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p


@pytest.mark.parametrize("family", ["native", "vector"])
@pytest.mark.parametrize("emission", ["threshold", "softmax"])
def test_typed_target_reference(family: str, emission: Literal["threshold", "softmax"]):
    original = gp_root_recipe(family)
    domain = p.VariableDomain(
        "binary", ("no", "yes"), emission, (0.0,) if emission == "threshold" else ()
    )
    equation = p.FunctionalCausalNode(
        (0,),
        p.FrozenLinearFunction(
            ((2.0, -1.0),) if emission == "threshold" else ((2.0, -1.0), (-2.0, 1.0))
        ),
    )
    base = original.base
    if isinstance(base, p.NativeCausalRecipe):
        base = replace(
            base,
            task=replace(base.task, task_type="binary", classes=2),
            nodes=(base.nodes[0], p.NativeCausalNode(equation, domain)),
        )
    else:
        assert isinstance(base, p.NativeVectorRecipe)
        base = replace(
            base,
            task=replace(base.task, task_type="binary", classes=2),
            nodes=(base.nodes[0], p.NativeVectorNode(equation, (domain,))),
        )
    retained = replace(original, base=base)
    ref = p.evaluate_gp_root_reference(retained, row=3, inputs=((-1.0, 0.0), (1.0, 0.0)))
    assert ref.target_mean is None
    assert ref.classes == ("no", "yes")
    assert ref.probabilities is not None
    expected = (
        np.asarray([[1.0, 0.0], [0.0, 1.0]])
        if emission == "threshold"
        else np.asarray(
            [
                [1 / (1 + np.exp(4)), 1 / (1 + np.exp(-4))],
                [1 / (1 + np.exp(-4)), 1 / (1 + np.exp(4))],
            ]
        )
    )
    np.testing.assert_allclose(ref.probabilities, expected, atol=1e-14)
    assert all(type(label) is int for label in ref.targets)
    assert p.gp_root_reference_from_dict(json.loads(json.dumps(asdict(ref)))) == ref


@pytest.mark.parametrize("policy", ["capture", "representation_and_capture"])
def test_converter_category_cut_preserves_or_replaces_descendant_state(policy: str):
    task = p.LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    converter = p.CategoricalConverter(
        "kind", 0, 2, ("left", "right"), "neighbor_disc", ((-2.0, -3.0), (4.0, 5.0))
    )
    root = p.ConvertedCausalNode(
        p.FunctionalCausalNode(
            (), p.FrozenLinearFunction(((1.0,), (-1.0,))), p.JointRootBase("gaussian", 1)
        ),
        (converter,),
    )
    child = p.ConvertedCausalNode(
        p.FunctionalCausalNode((0,), p.FrozenLinearFunction(((2.0, 1.0),)))
    )
    base = p.ConvertedCausalRecipe(
        task,
        (root, child),
        (p.ConvertedReadout(0, capture="kind"),),
        p.ConvertedReadout(1, channel=0),
    )
    pop = p.realize_gp_columns(p.GPColumnRequest(rows=16, columns=1), seed=7, task_id=9)
    retained = p.GPRootDAGRecipe(pop, base, (p.GPRootBinding(0, (0,)),))
    ref = p.evaluate_gp_root_reference(
        retained, row=3, inputs=((0.0,), (1.0,)), capture_location=policy
    )
    if policy == "representation_and_capture":
        np.testing.assert_array_equal(ref.targets, (-7.0, 13.0))
    else:
        original = p.generate_task(retained, seed=7, task_id=9)
        np.testing.assert_array_equal(ref.targets, (float(original.targets[3]),) * 2)
    with pytest.raises(ValueError, match="integer|categor|code"):
        p.evaluate_gp_root_reference(retained, row=3, inputs=((0.5,),))
    assert p.gp_root_reference_from_dict(asdict(ref)) == ref


@pytest.mark.parametrize("kind", ["binary", "multiclass"])
def test_fixed_calibrated_class_adapter_literal(kind: str):

    original = gp_root_recipe("functional")
    assert isinstance(original.base, p.FunctionalCausalRecipe)
    task = replace(
        original.task, task_type=kind, classes=2 if kind == "binary" else 3, separation=0.7
    )
    child = p.FunctionalCausalNode(
        (0,),
        p.FrozenLinearFunction(
            ((2.0, -1.0),) if kind == "binary" else ((2.0, -1.0), (-1.0, 2.0), (1.0, 1.0))
        ),
    )
    base = replace(
        original.base,
        task=task,
        nodes=(original.base.nodes[0], child),
        target=((1, 0),) if kind == "binary" else ((1, 0), (1, 1), (1, 2)),
    )
    calibrated = p.calibrate_gp_root_dag(
        replace(original, base=base),
        calibration_rows=32,
        calibration_seed=101,
        calibration_task_id=103,
    )
    query = p.realize_gp_root_calibration_query(calibrated, seed=107, task_id=109)
    ref = p.evaluate_gp_root_reference(query, row=9, inputs=((1.0, -2.0), (3.0, 0.5)))
    raw = np.asarray([[4.0], [5.5]] if kind == "binary" else [[4.0, -5.0, -1.0], [5.5, -2.0, 3.5]])
    scores = (raw - np.asarray(calibrated.calibration.center)) / np.asarray(
        calibrated.calibration.scale
    )
    np.testing.assert_allclose(ref.raw_scores, raw, atol=1e-14)
    np.testing.assert_allclose(ref.scores, scores, atol=1e-14)
    if kind == "binary":
        positive = 1 / (1 + np.exp(-0.7 * scores[:, 0]))
        probabilities = np.column_stack((1 - positive, positive))
    else:
        exponents = np.exp(0.7 * scores - np.max(0.7 * scores, axis=1, keepdims=True))
        probabilities = exponents / exponents.sum(axis=1, keepdims=True)
    assert ref.probabilities is not None
    np.testing.assert_allclose(ref.probabilities, probabilities, atol=1e-14)
    key = json.dumps(
        ("functional-causal-v1", 107, 109, ("labels",)), sort_keys=True, separators=(",", ":")
    )
    generator = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16))
    uniform = float(generator.uniform(size=16)[9])
    expected_labels = tuple(
        sum(uniform >= float(value) for value in row[:-1]) for row in probabilities.cumsum(axis=1)
    )
    np.testing.assert_array_equal(ref.targets, expected_labels)
    assert p.gp_root_reference_from_dict(asdict(ref)) == ref
