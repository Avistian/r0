"""Target adapters and genuine category cuts remain distinct on retained GP roots."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from gp_root_fixture import gp_root_recipe

from r0.modules import priors as p
from r0.modules.generation import load_causal_pair, save_causal_pair


@pytest.mark.parametrize("kind", ["binary", "multiclass"])
@pytest.mark.parametrize("mode", ["matched", "population"])
def test_gp_intervention_class_adapter_keeps_fit_and_keyed_label(
    kind: str,
    mode: str,
    tmp_path: Path,
):
    original = gp_root_recipe("functional")
    assert isinstance(original.base, p.FunctionalCausalRecipe)
    width = 1 if kind == "binary" else 3
    controls = replace(
        original.task,
        task_type=kind,
        classes=2 if kind == "binary" else 3,
        separation=2.1,
        class_offsets=(0.3,) if width == 1 else (-0.3, 0.5, 1.1),
    )
    matrix = ((1.0, -1.0), (0.5, 0.2), (-1.0, 2.0))[:width]
    graph = replace(
        original.base,
        task=controls,
        nodes=(
            original.base.nodes[0],
            replace(original.base.nodes[1], function=p.FrozenLinearFunction(matrix)),
        ),
        target=tuple((1, i) for i in range(width)),
    )
    original = replace(original, base=graph)
    base = p.calibrate_gp_root_dag(
        original, calibration_rows=32, calibration_seed=61, calibration_task_id=67
    )
    new_matrix = ((-1.0, 2.0), (0.7, -0.4), (3.0, 0.2))[:width]
    change = p.CausalInterventionRecipe(
        graph,
        (p.SoftIntervention(1, function=p.FrozenLinearFunction(new_matrix)),),
        mode=mode,
        population=11,
    )
    pair = p.generate_gp_root_pair(base, change)
    after = p.recipe_from_dict(json.loads(pair.after.recipe_json)["recipe"])
    assert isinstance(after, p.GPRootIntervention)
    raw = np.asarray(after.population.draw.values)[:, :2] @ np.asarray(new_matrix).T
    scores = (raw - np.asarray(base.calibration.center)) / np.asarray(base.calibration.scale)
    logits = 2.1 * scores + np.asarray(controls.class_offsets)
    if kind == "binary":
        positive = 1 / (1 + np.exp(-logits[:, 0]))
        probabilities = np.column_stack((1 - positive, positive))
    else:
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exps / exps.sum(axis=1, keepdims=True)
    assert pair.after.probabilities is not None
    np.testing.assert_allclose(pair.after.probabilities.numpy(), probabilities, atol=1e-12, rtol=0)
    keys = ("labels",) if mode == "matched" else ("intervention-population-v1", 11, "labels")
    key = json.dumps(("functional-causal-v1", 47, 59, keys), sort_keys=True, separators=(",", ":"))
    uniform = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16)).uniform(
        size=16
    )
    np.testing.assert_array_equal(
        pair.after.targets, (uniform[:, None] >= probabilities.cumsum(axis=1)[:, :-1]).sum(axis=1)
    )
    saved = tmp_path / "pair"
    save_causal_pair(saved, pair, max_bytes=32_000_000)
    loaded = load_causal_pair(saved)
    assert loaded.after.recipe_json == pair.after.recipe_json
    assert torch.equal(loaded.after.targets, pair.after.targets)


@pytest.mark.parametrize("family", ["native", "vector"])
def test_gp_native_category_cut_is_not_a_numeric_target_adapter(family: str):
    original = gp_root_recipe("functional")
    assert isinstance(original.base, p.FunctionalCausalRecipe)
    root = original.base.nodes[0]
    child = replace(
        original.base.nodes[1],
        function=p.FrozenLinearFunction(((1.0, -1.0), (-1.0, 1.0), (0.5, 0.5))),
    )
    domain = p.VariableDomain("categorical", ("a", "b", "c"), "softmax")
    controls = replace(original.task, task_type="multiclass", classes=3)
    if family == "native":
        graph = p.NativeCausalRecipe(
            controls,
            (p.NativeCausalNode(root), p.NativeCausalNode(child, domain)),
            ((0, 0), (0, 1)),
            (1, 0),
        )
        changes = (p.HardIntervention(1, category="b"),)
    else:
        graph = p.NativeVectorRecipe(
            controls,
            (
                p.NativeVectorNode(root, (p.VariableDomain(),) * 2),
                p.NativeVectorNode(child, (domain,)),
            ),
            ((0, 0), (0, 1)),
            (1, 0),
        )
        changes = (p.NativeVectorHardIntervention(1, 0, category="b"),)
    base = replace(original, base=graph)
    pair = p.generate_gp_root_pair(base, p.CausalInterventionRecipe(graph, changes))
    assert torch.all(pair.after.targets == 1)
    assert pair.after.target_origin == "native-variable-v1"
    assert pair.after.classes == ("a", "b", "c")
    np.testing.assert_array_equal(pair.after.probabilities, np.tile([0, 1, 0], (16, 1)))
