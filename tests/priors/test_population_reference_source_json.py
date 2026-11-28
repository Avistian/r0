"""Serialized reference custody is canonical and unambiguous before source replay."""

import json
from dataclasses import asdict
from typing import Any

import pytest

from r0.modules import generation as g
from r0.modules import priors as p
from r0.modules.explorer import ExplorerSession


@pytest.fixture(scope="module")
def receipt(tmp_path_factory: pytest.TempPathFactory):
    catalog = ExplorerSession(tmp_path_factory.mktemp("reference-source")).request(
        "construction_examples", {}
    )["examples"]
    controls = next(
        value
        for name, value in catalog.items()
        if name.startswith("Population DAG: functional linear")
    )
    source = g.GraphConstruction(json.dumps(controls), 17).build(
        p.CausalLimits(), episode_seed=31, task_id=2**55 + 7
    )
    assert isinstance(source, p.RecursivePopulationGraph)
    reference = p.evaluate_population_graph_reference(source, row=7, feature=1, values=(-2.0, 2.0))
    return reference, type(source.bindings[0].sample.initial)


def test_duplicate_source_seed_precedes_restore(receipt: Any, monkeypatch: pytest.MonkeyPatch):
    reference, snapshot = receipt
    raw = asdict(reference)
    raw["source_json"] = '{"seed":999,' + raw["source_json"][1:]

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("ambiguous serialized source reached population replay")

    monkeypatch.setattr(snapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="source|canonical|duplicate"):
        p.population_graph_reference_from_dict(raw)


@pytest.mark.parametrize(
    "variant",
    [
        "equal-duplicate",
        "nested-duplicate",
        "leading-space",
        "pretty",
        "key-order",
        "escaped-key",
        "NaN",
        "Infinity",
        "-Infinity",
        "overflow",
        "array",
    ],
)
def test_noncanonical_source_rejects_before_replay(
    receipt: Any, monkeypatch: pytest.MonkeyPatch, variant: str
):
    reference, snapshot = receipt
    raw = asdict(reference)
    text = raw["source_json"]
    if variant == "equal-duplicate":
        text = '{"seed":31,' + text[1:]
    elif variant == "nested-duplicate":
        text = text.replace('"request":{', '"request":{"version":"conflicting",', 1)
    elif variant == "leading-space":
        text = " " + text
    elif variant == "pretty":
        text = json.dumps(json.loads(text), indent=2)
    elif variant == "key-order":
        text = json.dumps(dict(reversed(list(json.loads(text).items()))), separators=(",", ":"))
    elif variant == "escaped-key":
        text = text.replace('"seed":', '"\\u0073eed":', 1)
    elif variant == "array":
        text = "[]"
    else:
        constant = "1e999" if variant == "overflow" else variant
        text = '{"extra":' + constant + "," + text[1:]
    assert text != raw["source_json"]
    raw["source_json"] = text

    def forbidden(*args: Any, **kwargs: Any):
        pytest.fail("noncanonical serialized source reached population replay")

    monkeypatch.setattr(snapshot, "restore", forbidden)
    with pytest.raises(ValueError, match="source|canonical|duplicate|JSON"):
        p.population_graph_reference_from_dict(raw)


def test_canonical_source_full_reference_is_unchanged(receipt: Any):
    reference, _ = receipt
    restored = p.population_graph_reference_from_dict(json.loads(json.dumps(asdict(reference))))
    assert restored == reference
    assert restored.source_json == reference.source_json
