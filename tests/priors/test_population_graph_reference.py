"""Original-row causal responses use literal retained trees, not a shared row surrogate."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


def population(tmp_path: Path, family: str, equation: str = "Conv"):
    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    controls = next(
        v for k, v in examples.items() if k.startswith(f"Population DAG: {family} {equation}")
    )
    source = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(source, p.RecursivePopulationGraph)
    return source


def tree_value(forest: p.RecursiveForest, values: tuple[float, ...]) -> float:
    leaves: list[float] = []
    for tree in forest.trees:
        index = 0
        while tree.nodes[index].left != -1:
            node = tree.nodes[index]
            index = node.left if values[node.feature] <= node.threshold else node.right
        leaves.append(tree.nodes[index].value[0])
    return sum(leaves) / len(leaves)


@pytest.mark.parametrize("family", ["functional", "native", "vector", "converted"])
@pytest.mark.parametrize("equation", ["linear", "Conv", "network"])
def test_four_families_literal_original_row_tree_and_reference_archive(
    tmp_path: Path, family: str, equation: str
):
    source = population(tmp_path, family, equation)
    original = p.generate_task(source, seed=source.seed, task_id=source.task_id)
    row = 7
    grid = (-2.0, 0.0, 2.0)
    result = p.evaluate_population_graph_reference(source, row=row, feature=1, values=grid)
    binding = source.bindings[0]
    expected = tuple(
        (3 * tree_value(binding.sample.forests[row], (binding.inputs[row][0], v)),) for v in grid
    )
    assert result.scores == expected
    assert result.targets == tuple(v[0] for v in expected)
    assert json.loads(result.source_json) == json.loads(original.recipe_json)["recipe"]
    assert result.task_id == 2**55 + 7
    assert p.population_graph_reference_from_dict(json.loads(json.dumps(asdict(result)))) == result
    anchor = p.evaluate_population_graph_reference(
        source, row=row, feature=1, values=(float(original.inputs[row, 1]),)
    )
    assert anchor.scores == (tuple(float(v) for v in original.scores[row]),)
    assert (
        original.recipe_json
        == p.generate_task(source, seed=source.seed, task_id=source.task_id).recipe_json
    )


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_retained_population_stream_and_unrelated_cut_are_preserved(tmp_path: Path, mode: str):
    source = population(tmp_path, "functional")
    change = p.CausalInterventionRecipe(
        source.request.base,
        (p.HardIntervention(2, (0,), (8.0,), location="state_and_readout"),),
        mode=mode,
        population=7,
    )
    changed = p.intervene_recursive_population_graph(source, change)
    result = p.evaluate_population_graph_reference(changed, row=7, feature=1, values=(-2.0, 2.0))
    assert result.scores == ((8.0,), (8.0,))
    assert result.population_mode == mode
    assert result.population_index == (7 if mode == "population" else None)
    assert json.loads(result.source_json) == json.loads(json.dumps(asdict(changed)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row", True),
        ("row", 16),
        ("feature", -1),
        ("values", (float("nan"),)),
        ("values", (1,)),
        ("values", (0.0,) * 130),
        ("capture_location", "implicit"),
        ("limits", p.CausalLimits(working_bytes=1)),
    ],
)
def test_invalid_controls_and_caps_precede_every_snapshot_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    source = population(tmp_path, "functional")

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("invalid reference controls must reject before any construction replay")

    monkeypatch.setattr(type(source.bindings[0].sample.initial), "restore", forbidden)
    controls: dict[str, Any] = dict(row=7, feature=1, values=(-2.0, 2.0))
    controls[field] = value
    with pytest.raises(ValueError, match="reference|limit|metadata"):
        p.evaluate_population_graph_reference(source, **controls)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("task_id", 2),
        ("row", -1),
        ("population_mode", "new"),
        ("population_index", False),
        ("scores", [[0.0]]),
        ("probabilities", [[1.0], [1.0]]),
        ("targets", [False, False]),
        ("anchor", [0]),
        ("semantics", "conditional-mean"),
        ("source_fingerprint", "forged"),
    ],
)
def test_reference_outer_metadata_rejects_before_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
):
    source = population(tmp_path, "functional")
    result = p.evaluate_population_graph_reference(source, row=7, feature=1, values=(-2.0, 2.0))
    raw = json.loads(json.dumps(asdict(result)))
    raw[field] = value

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("invalid reference receipt must reject before construction replay")

    monkeypatch.setattr(type(source.bindings[0].sample.initial), "restore", forbidden)
    with pytest.raises(ValueError, match="reference|limit|metadata"):
        p.population_graph_reference_from_dict(raw)


@pytest.mark.parametrize("mode", ["matched", "population"])
def test_soft_replacement_disables_population_but_preserves_equation(tmp_path: Path, mode: str):
    source = population(tmp_path, "functional")
    changed = p.intervene_recursive_population_graph(
        source,
        p.CausalInterventionRecipe(
            source.request.base,
            (p.SoftIntervention(1, function=p.FrozenLinearFunction(((2.0, 3.0),))),),
            mode=mode,
            population=7,
        ),
    )
    result = p.evaluate_population_graph_reference(changed, row=7, feature=1, values=(-2.0, 2.0))
    expected = tuple((3 * (2 * result.anchor[0] + 3 * v),) for v in result.values)
    assert result.scores == expected


@pytest.mark.parametrize("family", ["native", "vector"])
def test_classifier_forests_keep_literal_row_majority_and_indicator_probabilities(
    tmp_path: Path, family: str
):
    examples = ExplorerSession(tmp_path).request("construction_examples", {})["examples"]
    controls = examples[f"Population DAG: {family} majority-vote labels"]
    source = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(source, p.RecursivePopulationGraph)
    result = p.evaluate_population_graph_reference(
        source, row=7, feature=1, values=(-2.0, 0.0, 2.0)
    )
    winners: list[int] = []
    for value in result.values:
        inputs = (result.anchor[0], value)
        labels: list[int] = []
        for tree in source.bindings[0].sample.forests[7].trees:
            index = 0
            while tree.nodes[index].left != -1:
                node = tree.nodes[index]
                index = node.left if inputs[node.feature] <= node.threshold else node.right
            label = tree.nodes[index].label
            assert label is not None
            labels.append(label)
        winners.append(max(range(3), key=lambda label: labels.count(label)))
    assert result.targets == tuple(winners)
    assert result.probabilities == tuple(
        tuple(float(j == label) for j in range(3)) for label in winners
    )
    assert result.target_classes == ("a", "b", "c")


def test_observed_hard_cut_precedence_and_category_input_semantics(tmp_path: Path):
    source = population(tmp_path, "functional")
    changed = p.intervene_recursive_population_graph(
        source,
        p.CausalInterventionRecipe(
            source.request.base,
            (p.HardIntervention(0, (1,), (99.0,), location="state_and_readout"),),
        ),
    )
    result = p.evaluate_population_graph_reference(changed, row=7, feature=1, values=(-2.0, 2.0))
    forest = changed.samples[0].forests[7]
    assert result.scores == tuple(
        (3 * tree_value(forest, (result.anchor[0], v)),) for v in result.values
    )
    converted = population(tmp_path / "converted", "converted")
    capture = p.evaluate_population_graph_reference(converted, row=7, feature=0, values=(0.0, 1.0))
    assert capture.scores[0] == capture.scores[1]
    assert capture.vocabulary == ("low", "high")
    represented = p.evaluate_population_graph_reference(
        converted,
        row=7,
        feature=0,
        values=(0.0, 1.0),
        capture_location="representation_and_capture",
    )
    assert represented.capture_location == "representation_and_capture"


@pytest.mark.parametrize("binary", [False, True])
def test_fixed_original_adapter_noise_and_binary_probability(tmp_path: Path, binary: bool):
    original = population(tmp_path, "functional")
    task = replace(
        original.task,
        task_type="binary" if binary else "regression",
        classes=2 if binary else 0,
        noise_ratio=0.0 if binary else 0.4,
        separation=2.3 if binary else 1.0,
        class_offsets=(0.2,) if binary else (),
    )
    request = replace(original.request, base=replace(original.request.base, task=task))
    source = p.realize_recursive_population_graph(request, seed=31, task_id=2**55 + 7)
    realized = p.generate_task(source, seed=source.seed, task_id=source.task_id)
    result = p.evaluate_population_graph_reference(
        source, row=7, feature=1, values=(-2.0, 0.0, 2.0)
    )
    if binary:
        assert result.probabilities is not None
        for score, probabilities in zip(result.scores, result.probabilities, strict=True):
            expected = 1 / (1 + math.exp(-(2.3 * score[0] + 0.2)))
            assert probabilities == pytest.approx((1 - expected, expected), abs=1e-15)
    else:
        residual = float(realized.targets[7] - realized.scores[7, 0])
        for score, target in zip(result.scores, result.targets, strict=True):
            assert target - score[0] == pytest.approx(residual, abs=1e-15)
